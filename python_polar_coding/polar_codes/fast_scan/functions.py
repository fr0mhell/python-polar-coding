import numba
import numpy as np


@numba.njit
def compute_repetition_beta(alpha) -> np.array:
    """Compute beta value for Repetition node."""
    alpha_sum = np.sum(alpha)
    return -1 * alpha + alpha_sum


@numba.njit
def compute_spc_beta(alpha) -> np.array:
    """Compute beta value for Single parity node."""
    all_sign = np.sign(np.prod(alpha))
    abs_alpha = np.fabs(alpha)
    first_min_idx, second_min_idx = np.argsort(abs_alpha)[:2]

    result = np.sign(alpha) * all_sign
    for i in range(result.size):
        if i == first_min_idx:
            result[i] *= abs_alpha[second_min_idx]
        else:
            result[i] *= abs_alpha[first_min_idx]

    return result
