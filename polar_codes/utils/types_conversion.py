import numpy as np


def int_to_bin_list(value: int, size: int, as_array=True):
    """Get binary representation in a list form of given value.

    Args:
        value (int): value for binary representation.
        size (int): size of binary representation.
        as_array (bool):

    Returns:
        (list): binary representation of given value and size.

    """
    binary_list = [int(bit) for bit in np.binary_repr(value, width=size)]

    if as_array:
        return np.array(binary_list)
    return binary_list


def reverse_bits(value: int, size: int) -> int:
    """Reverse bits of n-bit integer value."""
    return int(''.join(reversed(np.binary_repr(value, width=size))), 2)
