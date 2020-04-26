from python_polar_coding.polar_codes.g_fast_ssc import (
    GeneralizedFastSSCPolarCodec,
)

from .decoder import EGFastSSCDecoder


class EGFastSSCPolarCodec(GeneralizedFastSSCPolarCodec):
    """Extended Generalized Fast SSC codec."""
    decoder_class = EGFastSSCDecoder
