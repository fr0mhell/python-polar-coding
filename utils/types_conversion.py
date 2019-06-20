import numpy as np


def int_to_bin_list(value: int, size: int, as_array=False):
    """Get binary representation in a list form of given value.

    Args:
        value (int): value for binary representation.
        size (int): size of binary representation.
        as_array (bool):

    Returns:
        (list): binary representation of given value and size.

    """
    if as_array:
        return np.array([int(bit) for bit in bin(value)[2:].zfill(size)])
    return [int(bit) for bit in bin(value)[2:].zfill(size)]
