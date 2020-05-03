import numba
import numpy as np

from ..fast_scan import compute_spc_beta
from ..rc_scan import compute_beta_one_node


@numba.njit
def compute_g_repetition(llr, mask_steps, last_chunk_type, N):
    """Compute bits for Generalized Repetition node.

    Based on: https://arxiv.org/pdf/1804.09508.pdf, Section III, A.

    """
    step = N // mask_steps  # step is equal to a chunk size

    last_alpha = np.zeros(step)
    for i in range(step):
        last_alpha[i] = np.sum(np.array([
            llr[i + j * step] for j in range(mask_steps)
        ]))

    last_beta = (
        compute_beta_one_node(last_alpha) if last_chunk_type == 1
        else compute_spc_beta(last_alpha)
    )

    result = np.zeros(N)
    for i in range(0, N, step):
        result[i: i + step] = last_beta

    return result


@numba.njit
def compute_rg_parity(llr, mask_steps, N):
    """Compute bits for Relaxed Generalized Parity Check node.

    Based on: https://arxiv.org/pdf/1804.09508.pdf, Section III, B.

    """
    step = N // mask_steps  # step is equal to a chunk size
    result = np.zeros(N)

    for i in range(step):
        alpha = np.zeros(mask_steps)
        for j in range(mask_steps):
            alpha[j] = llr[i + j * step]

        beta = compute_spc_beta(alpha)
        result[i:N:step] = beta

    return result
