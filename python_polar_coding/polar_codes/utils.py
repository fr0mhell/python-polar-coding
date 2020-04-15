import numpy as np
from numba import njit


def bitstring_to_bytes(s):
    """Converts bit string into bytes."""
    return int(s, 2).to_bytes(len(s) // 8, byteorder='big')


def int_to_bin_array(value: int, size: int) -> np.array:
    """Get binary representation in a list form of given value.

    Args:
        value (int): value for binary representation.
        size (int): size of binary representation.

    Returns:
        (list): binary representation of given value and size.

    """
    return np.array([int(bit) for bit in np.binary_repr(value, width=size)])


def reverse_bits(value: int, size: int) -> int:
    """Reverse bits of n-bit integer value."""
    return int(''.join(reversed(np.binary_repr(value, width=size))), 2)


@njit
def lowerconv(upperdecision: int, upperllr: float, lowerllr: float) -> float:
    """PERFORMS IN LOG DOMAIN
    llr = lowerllr * upperllr, if uppperdecision == 0
    llr = lowerllr / upperllr, if uppperdecision == 1

    """
    if upperdecision == 0:
        return lowerllr + upperllr
    else:
        return lowerllr - upperllr


@njit
def logdomain_sum(x: float, y: float) -> float:
    """"""
    if x < y:
        return y + np.log(1 + np.exp(x - y))
    else:
        return x + np.log(1 + np.exp(y - x))


@njit
def upperconv(llr1: float, llr2: float) -> float:
    """PERFORMS IN LOG DOMAIN
    llr = (llr1 * llr2 + 1) / (llr1 + llr2)

    """
    return logdomain_sum(llr1 + llr2, 0) - logdomain_sum(llr1, llr2)


def splits(start, end):
    while start <= end:
        yield start
        start *= 2
