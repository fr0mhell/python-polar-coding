import numpy as np
from anytree import PreOrderIter

from python_polar_coding.polar_codes.base import BaseTreeDecoder
from python_polar_coding.polar_codes.base.functions import make_hard_decision

from .functions import (
    compute_left_alpha,
    compute_parent_beta,
    compute_right_alpha,
)
from .node import RCSCANNode


class RCSCANDecoder(BaseTreeDecoder):
    """Implements Reduced-complexity SCAN decoding algorithm.

    Based on:
        * https://arxiv.org/pdf/1510.06495.pdf
        * doi:10.1007/s12243-018-0634-7

    """
    node_class = RCSCANNode

    def __init__(
            self,
            n: int,
            mask: np.array,
            I: int = 1,
    ):
        super().__init__(n=n, mask=mask)
        self.I = I

    def decode(self, received_llr: np.array) -> np.array:
        """Implementation of SC decoding method."""
        self._clean_before_decoding()

        for leaf in self.leaves:
            leaf.initialize_leaf_beta()

        for _ in range(self.I):
            super().decode(received_llr)

        return self.result

    def _clean_before_decoding(self):
        """Reset intermediate BETA values.

        Run this before calling `__call__` method.

        """
        for node in PreOrderIter(self._decoding_tree):
            if not (node.is_zero or node.is_one):
                node.beta *= 0

    def _compute_intermediate_alpha(self, leaf):
        """Compute intermediate Alpha values (LLR)."""
        for node in leaf.path[1:]:
            if node.is_computed or node.is_zero or node.is_one:
                continue

            parent_alpha = node.parent.alpha

            if node.is_left:
                right_beta = node.siblings[0].beta
                node.alpha = compute_left_alpha(parent_alpha, right_beta)

            if node.is_right:
                left_beta = node.siblings[0].beta
                node.alpha = compute_right_alpha(parent_alpha, left_beta)

            node.is_computed = True

    def _compute_intermediate_beta(self, node):
        """Compute intermediate BETA values."""
        parent = node.parent
        if node.is_left or node.is_root or parent.is_root:
            return

        left = node.siblings[0]
        parent.beta = compute_parent_beta(left.beta, node.beta, parent.alpha)
        return self._compute_intermediate_beta(parent)

    @property
    def result(self):
        return make_hard_decision(self.root.alpha +
                                  self._compute_result_beta())

    def _compute_result_beta(self) -> np.array:
        """Compute result BETA values."""
        alpha = self.root.alpha
        if not self.root.children:
            return self.root.beta

        left, right = self.root.children
        return compute_parent_beta(left.beta, right.beta, alpha)
