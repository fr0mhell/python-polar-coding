import numba
import numpy as np

from ..base import INFINITY, compute_alpha


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
