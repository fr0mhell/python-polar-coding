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
def make_hard_decision(soft_input):
    """Makes hard decision based on soft input values (LLR)."""
    return np.array([s < 0 for s in soft_input], dtype=np.int8)
