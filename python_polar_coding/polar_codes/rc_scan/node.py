from typing import Dict

from python_polar_coding.polar_codes.base.functions.beta_soft import (
    compute_beta_soft,
    one,
    zero,
)

from ..base import NodeTypes, SoftNode


class RCSCANNode(SoftNode):
    supported_nodes = (
        NodeTypes.ZERO,
        NodeTypes.ONE,
    )

    @property
    def is_zero(self) -> bool:
        """Check is the node is Zero node."""
        return self.node_type == NodeTypes.ZERO

    @property
    def is_one(self) -> bool:
        """Check is the node is One node."""
        return self.node_type == NodeTypes.ONE

    def get_decoding_params(self) -> Dict:
        return dict(
            node_type=self.node_type,
            llr=self.alpha,
        )

    def compute_leaf_beta(self):
        """Do nothing for ZERO and ONE nodes.

        Unlike SC-based decoders SCAN decoders does not make decisions
        in leaves.

        """
        if self.is_one or self.is_zero:
            return
        self.beta = compute_beta_soft(self.node_type, self.alpha)

    def initialize_leaf_beta(self):
        """Initialize BETA values on tree building.

        Initialize ZERO and ONE nodes following to Section III
        doi:10.1109/jsac.2014.140515

        """
        if not self.is_leaf:
            return

        if self.is_zero:
            self._beta = zero(self.alpha)
        if self.is_one:
            self._beta = one(self.alpha)
