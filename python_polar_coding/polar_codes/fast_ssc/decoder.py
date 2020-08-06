from python_polar_coding.polar_codes.base import BaseTreeDecoder
from python_polar_coding.polar_codes.base.functions import (
    compute_left_alpha,
    compute_parent_beta_hard,
    compute_right_alpha,
)

from .node import FastSSCNode


class FastSSCDecoder(BaseTreeDecoder):
    """Implements Fast SSC decoding algorithm."""
    node_class = FastSSCNode

    def _compute_intermediate_alpha(self, leaf):
        """Compute intermediate Alpha values (LLR)."""
        for node in leaf.path[1:]:
            if node.is_computed:
                continue

            # No need to compute zero node because output is vector of zeros
            if node.is_zero:
                continue

            parent_alpha = node.parent.alpha

            if node.is_left:
                node.alpha = compute_left_alpha(parent_alpha)
                continue

            left_node = node.siblings[0]
            left_beta = left_node.beta
            node.alpha = compute_right_alpha(parent_alpha, left_beta)
            node.is_computed = True

    def _compute_intermediate_beta(self, node):
        """Compute intermediate Beta values (BIT)."""
        if node.is_left:
            return

        if node.is_root:
            return

        parent = node.parent
        left = node.siblings[0]
        parent.beta = compute_parent_beta_hard(left.beta, node.beta)
        return self._compute_intermediate_beta(parent)
