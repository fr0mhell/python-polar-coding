import numba
import numpy as np

from python_polar_coding.polar_codes.base.functions.alpha import (
    function_1,
    function_2,
)

from ..base import INFINITY


@numba.njit
def compute_beta_zero_node(alpha):
    """Compute beta values for ZERO node.

    https://arxiv.org/pdf/1510.06495.pdf Section III.C.

    """
    return np.ones(alpha.size, dtype=np.double) * INFINITY


@numba.njit
def compute_beta_one_node(alpha):
    """Compute beta values for ONE node.

    https://arxiv.org/pdf/1510.06495.pdf Section III.C.

    """
    return np.zeros(alpha.size, dtype=np.double)


@numba.njit
def compute_left_alpha(parent_alpha, beta):
    """Compute LLR for left node."""
    N = parent_alpha.size // 2
    left_parent_alpha = parent_alpha[:N]
    right_parent_alpha = parent_alpha[N:]

    return function_1(left_parent_alpha, right_parent_alpha, beta)


@numba.njit
def compute_right_alpha(parent_alpha, beta):
    """Compute LLR for right node."""
    N = parent_alpha.size // 2
    left_parent_alpha = parent_alpha[:N]
    right_parent_alpha = parent_alpha[N:]

    return function_2(left_parent_alpha, beta, right_parent_alpha)


@numba.njit
def compute_parent_beta(left_beta, right_beta, parent_alpha):
    """Compute bits of a parent Node."""
    N = parent_alpha.size // 2
    left_parent_alpha = parent_alpha[:N]
    right_parent_alpha = parent_alpha[N:]

    result = np.zeros(parent_alpha.size, dtype=np.double)

    result[:N] = function_1(left_beta, right_beta, right_parent_alpha)
    result[N:] = function_2(left_beta, left_parent_alpha, right_beta)

    return result
