import numpy as np


def bhattacharyya_bounds(codeword_length: int, design_snr: float):
    """Estimate Bhattacharyya bounds of bit channels of polar code."""
    bhattacharya_bounds = np.zeros(codeword_length, dtype=np.double)
    snr = np.power(10, design_snr / 10)
    bhattacharya_bounds[0] = np.exp(-snr)

    for level in range(1, int(np.log2(codeword_length)) + 1):
        B = np.power(2, level)

        for j in range(int(B / 2)):
            val = bhattacharya_bounds[j]
            # TODO: refactor with `logdomain_diff` same with Matlab model
            bhattacharya_bounds[j] = 2 * val - np.power(val, 2)
            bhattacharya_bounds[int(B / 2 + j)] = np.power(val, 2)

    return bhattacharya_bounds
