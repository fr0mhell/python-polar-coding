from typing import Dict

from python_polar_coding.polar_codes.fast_ssc import FastSSCNode

from ..base import NodeTypes


class GFastSSCNode(FastSSCNode):
    """Decoder for Generalized Fast SSC code.

    Based on: https://arxiv.org/pdf/1804.09508.pdf

    """
    supported_nodes = (
        NodeTypes.ZERO,
        NodeTypes.ONE,
        NodeTypes.SINGLE_PARITY_CHECK,
        NodeTypes.REPETITION,
        NodeTypes.RG_PARITY,
        NodeTypes.G_REPETITION,
    )

    def get_decoding_params(self) -> Dict:
        return dict(
            node_type=self.node_type,
            llr=self.alpha,
            mask_steps=self.mask_steps,
            last_chunk_type=self.last_chunk_type,
        )
