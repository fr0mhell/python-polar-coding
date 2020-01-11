"""Logarithmic base operations"""
import numpy as np


def lowerconv(upperdecision: int, upperllr: float, lowerllr: float) -> float:
    """PERFORMS IN LOG DOMAIN
    llr = lowerllr * upperllr, if uppperdecision == 0
    llr = lowerllr / upperllr, if uppperdecision == 1

    """
    if upperdecision == 0:
        return lowerllr + upperllr
    else:
        return lowerllr - upperllr


def logdomain_sum(x: float, y: float) -> float:
    """"""
    if x < y:
        return y + np.log(1 + np.exp(x - y))
    else:
        return x + np.log(1 + np.exp(y - x))


def upperconv(llr1: float, llr2: float) -> float:
    """PERFORMS IN LOG DOMAIN
    llr = (llr1 * llr2 + 1) / (llr1 + llr2)

    """
    return logdomain_sum(llr1 + llr2, 0) - logdomain_sum(llr1, llr2)
