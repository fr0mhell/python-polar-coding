from python_polar_coding.polar_codes.rc_scan import RCSCANNode

from ..base import NodeTypes


class FastSCANNode(RCSCANNode):
    supported_nodes = (
        NodeTypes.ZERO,
        NodeTypes.ONE,
        NodeTypes.REPETITION,
        NodeTypes.SINGLE_PARITY_CHECK,
    )
