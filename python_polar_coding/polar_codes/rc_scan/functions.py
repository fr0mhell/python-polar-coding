import numba

from ..base import compute_alpha


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
