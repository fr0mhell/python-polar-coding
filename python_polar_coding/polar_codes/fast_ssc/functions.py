import numba
import numpy as np

from ..base import make_hard_decision


@numba.njit
def compute_single_parity_check(llr):
    """Compute bits for Single Parity Check node.

    Based on: https://arxiv.org/pdf/1307.7154.pdf, Section IV, A.

    """
    bits = make_hard_decision(llr)
    parity = np.sum(bits) % 2
    arg_min = np.abs(llr).argmin()
    bits[arg_min] = (bits[arg_min] + parity) % 2
    return bits


@numba.njit
def compute_repetition(llr):
    """Compute bits for Repetition node.

    Based on: https://arxiv.org/pdf/1307.7154.pdf, Section IV, B.

    """
    return (
        np.zeros(llr.size, dtype=np.int8) if np.sum(llr) >= 0
        else np.ones(llr.size, dtype=np.int8)
    )
