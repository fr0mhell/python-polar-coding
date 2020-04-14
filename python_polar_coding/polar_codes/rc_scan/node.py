import numba
import numpy as np

from python_polar_coding.polar_codes.fast_ssc import FastSSCNode

# LLR = 1000 is high enough to be considered as +∞ for RC-SCAN decoding
INFINITY = 1000


class RCSCANNode(FastSSCNode):

    def compute_leaf_beta(self):
        """Do nothing.

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
            self._beta = self._compute_zero_node_beta(self.alpha)
        if self._node_type == RCSCANNode.ONE_NODE:
            self._beta = self._compute_one_node_beta(self.alpha)

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

    @staticmethod
    @numba.njit
    def _compute_zero_node_beta(llr):
        """Compute beta values for ZERO node.

        https://arxiv.org/pdf/1510.06495.pdf Section III.C.

        """
        return np.ones(llr.size, dtype=np.double) * INFINITY

    @staticmethod
    @numba.njit
    def _compute_one_node_beta(llr):
        """Compute beta values for ONE node.

        https://arxiv.org/pdf/1510.06495.pdf Section III.C.

        """
        return np.zeros(llr.size, dtype=np.double)
