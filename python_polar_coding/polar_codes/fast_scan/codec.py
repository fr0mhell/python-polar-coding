from python_polar_coding.polar_codes.rc_scan import RCSCANPolarCodec

from .decoder import FastSCANDecoder


class FastSCANCodec(RCSCANPolarCodec):
    decoder_class = FastSCANDecoder
