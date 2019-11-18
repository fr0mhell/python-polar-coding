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
def basic_llr_computation(a, b):
    """Basic function to compute intermediate LLR values."""
    return np.sign(a) * np.sign(b) * np.fabs(np.array([a, b])).min()


@numba.njit
def function_1(a, b, c):
    """Function 1.

    Source: doi:10.1007/s12243-018-0634-7, formula 1.

    """
    return basic_llr_computation(a, b + c)


@numba.njit
def function_2(a, b, c):
    """Function 2.

    Source: doi:10.1007/s12243-018-0634-7, formula 2.

    """
    return basic_llr_computation(a, b) + c


@numba.njit
def make_hard_decision(soft_input):
    """Makes hard decision based on soft input values (LLR)."""
    return np.array([s < 0 for s in soft_input], dtype=np.int8)
