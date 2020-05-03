from python_polar_coding.polar_codes.rc_scan import RCSCANDecoder

from .node import FastSCANNode


class FastSCANDecoder(RCSCANDecoder):
    node_class = FastSCANNode
