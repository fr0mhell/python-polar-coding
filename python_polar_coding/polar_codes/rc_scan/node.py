import numpy as np

from python_polar_coding.polar_codes.fast_ssc import FastSSCNode

from .functions import compute_beta_one_node, compute_beta_zero_node


class RCSCANNode(FastSSCNode):

    def compute_leaf_beta(self):
        """Do nothing for ZERO and ONE nodes.

        Unlike SC-based decoders SCAN decoders does not make decisions
        in leaves.

        """

    def initialize_leaf_beta(self):
        """Initialize BETA values on tree building.

        Initialize Leaves following to Section III doi:10.1109/jsac.2014.140515

        """
        if not self.is_leaf:
            return

        if self._node_type == RCSCANNode.ZERO_NODE:
            self._beta = compute_beta_zero_node(self.alpha)
        if self._node_type == RCSCANNode.ONE_NODE:
            self._beta = compute_beta_one_node(self.alpha)

    def get_node_type(self):
        """Get the type of RC SCAN Node.

        * Zero node - [0, 0, 0, 0, 0, 0, 0, 0];
        * One node - [1, 1, 1, 1, 1, 1, 1, 1];

        Or other type.

        """
        if np.all(self._mask == 0):
            return RCSCANNode.ZERO_NODE
        if np.all(self._mask == 1):
            return RCSCANNode.ONE_NODE
        return RCSCANNode.OTHER
