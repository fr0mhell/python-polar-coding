from python_polar_coding.polar_codes.rc_scan import (
    RCSCANNode,
    compute_beta_one_node,
    compute_beta_zero_node,
)

from .functions import compute_repetition_beta, compute_spc_beta


class FastSCANNode(RCSCANNode):

    def compute_leaf_beta(self):
        """Compute leaf beta."""
        if self.is_repetition:
            self.beta = compute_repetition_beta(self.alpha)
        if self.is_parity:
            self.beta = compute_spc_beta(self.alpha)

    def initialize_leaf_beta(self):
        """Initialize BETA values on tree building.

        Initialize ZERO and ONE nodes following to Section III
        doi:10.1109/jsac.2014.140515

        """
        if not self.is_leaf:
            return

        if self.is_parity or self.is_repetition:
            return

        if self.is_zero:
            self._beta = compute_beta_zero_node(self.alpha)
        if self.is_one:
            self._beta = compute_beta_one_node(self.alpha)

    def get_node_type(self):
        """Get the type of Fast SCAN Node.

        * Zero node - [0, 0, 0, 0, 0, 0, 0, 0];
        * One node - [1, 1, 1, 1, 1, 1, 1, 1];
        * Single parity check node - [0, 1, 1, 1, 1, 1, 1, 1];
        * Repetition node - [0, 0, 0, 0, 0, 0, 0, 1].

        Or other type.

        """
        if self._check_is_zero(self._mask):
            return self.ZERO_NODE
        if self._check_is_one(self._mask):
            return self.ONE_NODE
        if self._check_is_rep(self._mask):
            return self.REPETITION
        if self._check_is_spc(self._mask):
            return self.SINGLE_PARITY_CHECK
        return self.OTHER
