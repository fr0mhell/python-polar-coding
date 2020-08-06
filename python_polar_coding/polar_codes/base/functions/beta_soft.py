"""Common functions for polar coding."""
import numba
import numpy as np

from ..constants import INFINITY
from .node_types import NodeTypes


@numba.njit
def zero(
        llr: np.array,
        mask_steps: int = 0,
        last_chunk_type: int = 0,
) -> np.array:
    """Compute beta values for ZERO node.

    https://arxiv.org/pdf/1510.06495.pdf Section III.C.

    """
    return np.ones(llr.size, dtype=np.double) * INFINITY


@numba.njit
def one(
        llr: np.array,
        mask_steps: int = 0,
        last_chunk_type: int = 0,
) -> np.array:
    """Compute beta values for ONE node.

    https://arxiv.org/pdf/1510.06495.pdf Section III.C.

    """
    return np.zeros(llr.size, dtype=np.double)


@numba.njit
def repetition(
        llr: np.array,
        mask_steps: int = 0,
        last_chunk_type: int = 0,
) -> np.array:
    """Compute beta value for Repetition node."""
    alpha_sum = np.sum(llr)
    return -1 * llr + alpha_sum


@numba.njit
def single_parity_check(
        llr: np.array,
        mask_steps: int = 0,
        last_chunk_type: int = 0,
) -> np.array:
    """Compute beta value for Single parity node."""
    all_sign = np.sign(np.prod(llr))
    abs_alpha = np.fabs(llr)
    first_min_idx, second_min_idx = np.argsort(abs_alpha)[:2]

    result = np.sign(llr) * all_sign
    for i in range(result.size):
        if i == first_min_idx:
            result[i] *= abs_alpha[second_min_idx]
        else:
            result[i] *= abs_alpha[first_min_idx]

    return result


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
        one(last_alpha) if last_chunk_type == 1
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
    NodeTypes.ONE: one,
    NodeTypes.SINGLE_PARITY_CHECK: single_parity_check,
    NodeTypes.REPETITION: repetition,
    NodeTypes.RG_PARITY: rg_parity,
    NodeTypes.G_REPETITION: g_repetition,
}


def compute_beta_soft(
        node_type: str,
        llr: np.array,
        mask_steps: int = 0,
        last_chunk_type: int = 0,
        *args, **kwargs,
) -> np.array:
    """Unites functions for making soft decisions during decoding."""
    method = _methods_map[node_type]
    return method(llr, mask_steps, last_chunk_type, *args, **kwargs)
