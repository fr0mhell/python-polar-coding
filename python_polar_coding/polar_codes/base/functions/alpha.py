import numba
import numpy as np


@numba.njit
def compute_alpha(a: np.array, b: np.array) -> np.array:
    """Basic function to compute intermediate LLR values."""
    c = np.zeros(a.shape[0])
    for i in range(c.shape[0]):
        c[i] = (
            np.sign(a[i]) *
            np.sign(b[i]) *
            np.fabs(np.array([a[i], b[i]])).min()
        )
    return c


@numba.njit
def compute_left_alpha(llr: np.array) -> np.array:
    """Compute Alpha for left node during SC-based decoding."""
    N = llr.size // 2
    left = llr[:N]
    right = llr[N:]
    return compute_alpha(left, right)


@numba.njit
def compute_right_alpha(llr: np.array, left_beta: np.array) -> np.array:
    """Compute Alpha for right node during SC-based decoding."""
    N = llr.size // 2
    left = llr[:N]
    right = llr[N:]
    return right - (2 * left_beta - 1) * left


@numba.njit
def function_1(a: np.array, b: np.array, c: np.array) -> np.array:
    """Function 1.

    Source: doi:10.1007/s12243-018-0634-7, formula 1.

    """
    return compute_alpha(a, b + c)


@numba.njit
def function_2(a: np.array, b: np.array, c: np.array) -> np.array:
    """Function 2.

    Source: doi:10.1007/s12243-018-0634-7, formula 2.

    """
    return compute_alpha(a, b) + c
