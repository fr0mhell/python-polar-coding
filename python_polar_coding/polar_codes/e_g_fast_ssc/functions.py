import numba
import numpy as np

from ..base import compute_left_alpha


@numba.njit
def compute_left_alpha_sign(alpha: np.array) -> np.array:
    """"""
    left_alpha = compute_left_alpha(alpha)
    return np.sign(np.sum(left_alpha))


@numba.njit
def compute_right_alpha(alpha: np.array, left_sign: int = 1) -> np.array:
    """`left_sign` is 1 or -1"""
    N = alpha.size // 2
    left_alpha = alpha[:N]
    right_alpha = alpha[N:]
    return right_alpha + left_alpha * left_sign
