from python_polar_coding.polar_codes.fast_scan import (
    compute_repetition_beta,
    compute_spc_beta,
)
from python_polar_coding.polar_codes.g_fast_ssc import GeneralizedFastSSCNode
from python_polar_coding.polar_codes.rc_scan import (
    compute_beta_one_node,
    compute_beta_zero_node,
)

from .functions import compute_g_repetition, compute_rg_parity


class GFastSCANNode(GeneralizedFastSSCNode):

    def compute_leaf_beta(self):
        """Do nothing for ZERO and ONE nodes.

        Unlike SC-based decoders SCAN decoders does not make decisions
        in ZERO and ONE leaves.

        """
        if self.is_repetition:
            self.beta = compute_repetition_beta(self.alpha)
        if self.is_parity:
            self.beta = compute_spc_beta(self.alpha)
        if self.is_g_repetition:
            self.beta = compute_g_repetition(
                llr=self.alpha,
                mask_steps=self.mask_steps,
                last_chunk_type=self.last_chunk_type,
                N=self.N,
            )
        if self.is_rg_parity:
            self.beta = compute_rg_parity(
                llr=self.alpha,
                mask_steps=self.mask_steps,
                N=self.N,
            )

    def initialize_leaf_beta(self):
        """Initialize BETA values on tree building.

        Initialize ZERO and ONE nodes following to Section III
        doi:10.1109/jsac.2014.140515

        """
        if not self.is_leaf:
            return

        if self.is_zero:
            self._beta = compute_beta_zero_node(self.alpha)
        if self.is_one:
            self._beta = compute_beta_one_node(self.alpha)
