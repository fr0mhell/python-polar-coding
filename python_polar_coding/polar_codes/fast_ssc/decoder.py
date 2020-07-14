import numpy as np
from anytree import PreOrderIter

from python_polar_coding.polar_codes.base.functions import (
    compute_left_alpha,
    compute_right_alpha,
    compute_parent_beta_hard,
)
from python_polar_coding.polar_codes.sc import SCDecoder

from .node import FastSSCNode


class FastSSCDecoder(SCDecoder):
    """Implements Fast SSC decoding algorithm."""
    node_class = FastSSCNode

    def __init__(
            self,
            n: int,
            mask: np.array,
            is_systematic: bool = True,
            code_min_size: int = 0,
    ):
        super().__init__(n=n, mask=mask, is_systematic=is_systematic)
        self._decoding_tree = self.setup_decoding_tree(code_min_size)
        self._position = 0

    def setup_decoding_tree(self, N_min, **kwargs):
        """Setup decoding tree."""
        return self.node_class(mask=self.mask, N_min=N_min)

    def _set_initial_state(self, received_llr):
        """Initialize decoder with received message."""
        self.current_state = np.zeros(self.n, dtype=np.int8)
        self.previous_state = np.ones(self.n, dtype=np.int8)

        # LLR values at intermediate steps
        self._position = 0
        self._decoding_tree.root.alpha = received_llr

    def decode_internal(self, received_llr: np.array) -> np.array:
        """Implementation of SC decoding method."""
        self._set_initial_state(received_llr)

        # Reset the state of the tree before decoding
        for node in PreOrderIter(self._decoding_tree):
            node.is_computed = False

        for leaf in self._decoding_tree.leaves:
            self._set_decoder_state(self._position)
            self.compute_intermediate_alpha(leaf)
            leaf()
            self.compute_intermediate_beta(leaf)
            self.set_next_state(leaf.N)

        return self.result

    @property
    def root(self):
        """Returns root node of decoding tree."""
        return self._decoding_tree.root

    @property
    def result(self):
        if self.is_systematic:
            return self.root.beta

    def compute_intermediate_alpha(self, leaf):
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

    def compute_intermediate_beta(self, node):
        """Compute intermediate Beta values (BIT)."""
        if node.is_left:
            return

        if node.is_root:
            return

        parent = node.parent
        left = node.siblings[0]
        parent.beta = compute_parent_beta_hard(left.beta, node.beta)
        return self.compute_intermediate_beta(parent)

    def set_next_state(self, leaf_size):
        self._position += leaf_size
