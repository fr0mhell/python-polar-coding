from typing import Dict

from ..base import HardNode, NodeTypes


class FastSSCNode(HardNode):
    """Decoding node for Fast SSC algorithm."""
    supported_nodes = (
        NodeTypes.ZERO,
        NodeTypes.ONE,
        NodeTypes.SINGLE_PARITY_CHECK,
        NodeTypes.REPETITION,
    )

    @property
    def is_zero(self) -> bool:
        """Check is the node is Zero node."""
        return self.node_type == NodeTypes.ZERO

    def get_decoding_params(self) -> Dict:
        return dict(
            node_type=self.node_type,
            llr=self.alpha,
        )
