import numba
import numpy as np


@numba.njit
def compute_left_llr(llr):
    """Compute LLR for left node."""
    left_llr = np.zeros(llr.size // 2, dtype=np.double)
    for i in range(left_llr.size):
        left_llr[i] = (
            np.sign(llr[2 * i]) *
            np.sign(llr[2 * i + 1]) *
            np.fabs(llr[2 * i: 2 * i + 2]).min()
        )
    return left_llr


@numba.njit
def compute_right_llr(llr, left_bits):
    """Compute LLR for right node."""
    right_llr = np.zeros(llr.size // 2, dtype=np.double)
    for i in range(right_llr.size):
        right_llr[i] = (
            llr[2 * i + 1] -
            (2 * left_bits[i] - 1) * llr[2 * i]
        )
    return right_llr


@numba.njit
def compute_bits(left_bits, right_bits):
    """Compute intermediate bits."""
    result = np.zeros(int(left_bits.size * 2), dtype=np.int8)
    for i in range(left_bits.size):
        result[2 * i] = left_bits[i] ^ right_bits[i]
        result[2 * i + 1] = right_bits[i]
    return result
