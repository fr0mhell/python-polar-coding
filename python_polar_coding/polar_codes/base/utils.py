import numpy as np


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
