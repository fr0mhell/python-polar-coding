"""Common functions for polar coding."""
import numba
import numpy as np


@numba.njit
def compute_encoding_step(level, n, source, result):
    """Compute single step of polar encoding process."""
    pairs_per_group = step = np.power(2, n - level - 1)
    groups = np.power(2, level)

    for g in range(groups):
        start = 2 * g * step

        for p in range(pairs_per_group):
            result[p + start] = source[p + start] ^ source[p + start + step]
            result[p + start + step] = source[p + start + step]

    return result


@numba.njit
def compute_alpha(a, b):
    """Basic function to compute intermediate LLR values."""
    c = np.zeros(a.shape[0])
    for i in range(c.shape[0]):
        c[i] = np.sign(a[i]) * np.sign(b[i]) * np.fabs(np.array([a[i], b[i]])).min()
    return c


@numba.njit
def function_1(a, b, c):
    """Function 1.

    Source: doi:10.1007/s12243-018-0634-7, formula 1.

    """
    return compute_alpha(a, b + c)


@numba.njit
def function_2(a, b, c):
    """Function 2.

    Source: doi:10.1007/s12243-018-0634-7, formula 2.

    """
    return compute_alpha(a, b) + c


@numba.njit
def make_hard_decision(soft_input):
    """Makes hard decision based on soft input values (LLR)."""
    return np.array([s < 0 for s in soft_input], dtype=np.int8)


@numba.njit
def compute_single_parity_check(llr):
    """Compute bits for Single Parity Check node.

    Based on: https://arxiv.org/pdf/1307.7154.pdf, Section IV, A.

    """
    bits = make_hard_decision(llr)
    parity = np.sum(bits) % 2
    arg_min = np.abs(llr).argmin()
    bits[arg_min] = (bits[arg_min] + parity) % 2
    return bits


@numba.njit
def compute_repetition(llr):
    """Compute bits for Repetition node.

    Based on: https://arxiv.org/pdf/1307.7154.pdf, Section IV, B.

    """
    return (
        np.zeros(llr.size, dtype=np.int8) if np.sum(llr) >= 0
        else np.ones(llr.size, dtype=np.int8)
    )


@numba.njit
def compute_g_repetition(llr, mask_steps, last_chunk_type, N):
    """Compute bits for Generalized Repetition node.

    Based on: https://arxiv.org/pdf/1804.09508.pdf, Section III, A.

    """
    step = N // mask_steps  # step is equal to a chunk size

    last_alpha = np.zeros(step)
    for i in range(step):
        last_alpha[i] = np.sum(np.array([
            llr[i + j * step] for j in range(mask_steps)
        ]))

    last_beta = (
        make_hard_decision(last_alpha) if last_chunk_type == 1
        else compute_single_parity_check(last_alpha)
    )

    result = np.zeros(N)
    for i in range(0, N, step):
        result[i: i + step] = last_beta

    return result


@numba.njit
def compute_rg_parity(llr, mask_steps, N):
    """Compute bits for Relaxed Generalized Parity Check node.

    Based on: https://arxiv.org/pdf/1804.09508.pdf, Section III, B.

    """
    step = N // mask_steps  # step is equal to a chunk size
    result = np.zeros(N)

    for i in range(step):
        alpha = np.zeros(mask_steps)
        for j in range(mask_steps):
            alpha[j] = llr[i + j * step]

        beta = compute_single_parity_check(alpha)
        result[i:N:step] = beta

    return result
