"""Common functions for polar coding."""
import numba
import numpy as np

from .node_types import NodeTypes

# -----------------------------------------------------------------------------
# Making hard decisions during the decoding
# -----------------------------------------------------------------------------


@numba.njit
def zero(
        llr: np.array,
        mask_steps: int = 0,
        last_chunk_type: int = 0,
) -> np.array:
    """Makes hard decision based on soft input values (LLR)."""
    return np.zeros(llr.size, dtype=np.int8)


@numba.njit
def make_hard_decision(
        llr: np.array,
        mask_steps: int = 0,
        last_chunk_type: int = 0,
) -> np.array:
    """Makes hard decision based on soft input values (LLR)."""
    return np.array([s < 0 for s in llr], dtype=np.int8)


@numba.njit
def single_parity_check(
        llr: np.array,
        mask_steps: int = 0,
        last_chunk_type: int = 0,
) -> np.array:
    """Compute bits for Single Parity Check node.

    Based on: https://arxiv.org/pdf/1307.7154.pdf, Section IV, A.

    """
    bits = make_hard_decision(llr)
    parity = np.sum(bits) % 2
    arg_min = np.abs(llr).argmin()
    bits[arg_min] = (bits[arg_min] + parity) % 2
    return bits


@numba.njit
def repetition(
        llr: np.array,
        mask_steps: int = 0,
        last_chunk_type: int = 0,
) -> np.array:
    """Compute bits for Repetition node.

    Based on: https://arxiv.org/pdf/1307.7154.pdf, Section IV, B.

    """
    return (
        np.zeros(llr.size, dtype=np.int8) if np.sum(llr) >= 0
        else np.ones(llr.size, dtype=np.int8)
    )


@numba.njit
def g_repetition(
        llr: np.array,
        mask_steps: int,
        last_chunk_type: int,
) -> np.array:
    """Compute bits for Generalized Repetition node.

    Based on: https://arxiv.org/pdf/1804.09508.pdf, Section III, A.

    """
    N = llr.size
    step = N // mask_steps  # step is equal to a chunk size

    last_alpha = np.zeros(step)
    for i in range(step):
        last_alpha[i] = np.sum(np.array([
            llr[i + j * step] for j in range(mask_steps)
        ]))

    last_beta = (
        make_hard_decision(last_alpha) if last_chunk_type == 1
        else single_parity_check(last_alpha)
    )

    result = np.zeros(N)
    for i in range(0, N, step):
        result[i: i + step] = last_beta

    return result


@numba.njit
def rg_parity(
        llr: np.array,
        mask_steps: int,
        last_chunk_type: int = 0,
) -> np.array:
    """Compute bits for Relaxed Generalized Parity Check node.

    Based on: https://arxiv.org/pdf/1804.09508.pdf, Section III, B.

    """
    N = llr.size
    step = N // mask_steps  # step is equal to a chunk size
    result = np.zeros(N)

    for i in range(step):
        alpha = np.zeros(mask_steps)
        for j in range(mask_steps):
            alpha[j] = llr[i + j * step]

        beta = single_parity_check(alpha)
        result[i:N:step] = beta

    return result


# Mapping between decoding node types and corresponding decoding methods
_methods_map = {
    NodeTypes.ZERO: zero,
    NodeTypes.ONE: make_hard_decision,
    NodeTypes.SINGLE_PARITY_CHECK: single_parity_check,
    NodeTypes.REPETITION: repetition,
    NodeTypes.RG_PARITY: rg_parity,
    NodeTypes.G_REPETITION: g_repetition,
}


def compute_beta_hard(
        node_type: str,
        llr: np.array,
        mask_steps: int = 0,
        last_chunk_type: int = 0,
        *args, **kwargs,
) -> np.array:
    """Unites functions for making hard decisions during decoding."""
    method = _methods_map[node_type]
    return method(llr, mask_steps, last_chunk_type, *args, **kwargs)


@numba.njit
def compute_parent_beta_hard(left: np.array, right: np.array) -> np.array:
    """Compute Beta values for parent Node."""
    N = left.size
    result = np.zeros(N * 2, dtype=np.int8)
    result[:N] = (left + right) % 2
    result[N:] = right

    return result
