from python_polar_coding.polar_codes.g_fast_ssc import (
    GeneralizedFastSSCDecoder,
)

from .node import EGFastSSCNode


class EGFastSSCDecoder(GeneralizedFastSSCDecoder):
    """"""
    node_class = EGFastSSCNode
