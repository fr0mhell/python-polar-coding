import numba
import numpy as np


@numba.njit
def compute_encoding_step(
        level: int,
        n: int,
        source: np.array,
        result: np.array,
) -> np.array:
    """Compute single step of polar encoding process."""
    pairs_per_group = step = np.power(2, n - level - 1)
    groups = np.power(2, level)

    for g in range(groups):
        start = 2 * g * step

        for p in range(pairs_per_group):
            result[p + start] = source[p + start] ^ source[p + start + step]
            result[p + start + step] = source[p + start + step]

    return result
