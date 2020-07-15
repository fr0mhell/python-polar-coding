from typing import Dict

from python_polar_coding.polar_codes.fast_scan import FastSCANNode

from ..base import NodeTypes


class GFastSCANNode(FastSCANNode):
    supported_nodes = (
        NodeTypes.ZERO,
        NodeTypes.ONE,
        NodeTypes.REPETITION,
        NodeTypes.SINGLE_PARITY_CHECK,
        NodeTypes.G_REPETITION,
        NodeTypes.RG_PARITY,
    )

    def get_decoding_params(self) -> Dict:
        return dict(
            node_type=self.node_type,
            llr=self.alpha,
            mask_steps=self.mask_steps,
            last_chunk_type=self.last_chunk_type,
        )
