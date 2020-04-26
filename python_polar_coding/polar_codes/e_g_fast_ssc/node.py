import numpy as np

from python_polar_coding.polar_codes.g_fast_ssc import GeneralizedFastSSCNode

from .functions import compute_left_alpha_sign, compute_right_alpha


class EGFastSSCNode(GeneralizedFastSSCNode):
    """Decoder for Generalized Fast SSC code.

    Based on: https://arxiv.org/pdf/1804.09508.pdf

    """
    ZERO_ANY = 'ZERO-ANY'
    REP_ANY = 'REP-ANY'

    def __init__(self, *args, **kwargs):
        # Contains `ANY` node for ZERO_ANY or REP_ANY
        self.inner_node = None
        super().__init__(*args, **kwargs)

    @property
    def is_any(self):
        return (
            self.is_zero or
            self.is_one or
            self.is_repetition or
            self.is_parity or
            self.is_g_repetition or
            self.is_rg_parity
        )

    @property
    def is_zero_any(self):
        return self._node_type == self.ZERO_ANY

    @property
    def is_rep_any(self):
        return self._node_type == self.REP_ANY

    def get_node_type(self):
        ntype = super().get_node_type()
        if ntype != self.OTHER:
            return ntype
        if self._check_is_zero_any(self._mask):
            return self.ZERO_ANY
        if self._check_is_rep_any(self._mask):
            return self.REP_ANY
        return self.OTHER

    def _check_is_zero_any(self, mask):
        """"""
        left, right = np.split(mask, 2)
        if not self._check_is_zero(left):
            return False
        inner_node = self.__class__(
            mask=right,
            name=self.ROOT,
            N_min=self.N_min,
            AF=self.AF
        )
        if not inner_node.is_any:
            return False

        self.inner_node = inner_node
        return True

    def _check_is_rep_any(self, mask):
        """"""
        left, right = np.split(mask, 2)
        if not self._check_is_rep(left):
            return False
        right_node = self.__class__(
            mask=right,
            name=self.ROOT,
            N_min=self.N_min,
            AF=self.AF
        )
        if not right_node.is_any:
            return False

        self.inner_node = right_node
        return True

    def compute_leaf_beta(self):
        super().compute_leaf_beta()
        klass = self.__class__

        if self._node_type == klass.ZERO_ANY:
            self._beta = self.compute_zero_any()
        if self._node_type == klass.REP_ANY:
            self._beta = self.compute_rep_any()

    def compute_zero_any(self):
        """"""
        right_alpha = compute_right_alpha(self.alpha, left_sign=1)

        self.inner_node.alpha = right_alpha
        self.inner_node.compute_leaf_beta()

        beta = np.zeros(self.N, dtype=np.int8)
        beta[:self.inner_node.N] = self.inner_node.beta
        beta[self.inner_node.N:] = self.inner_node.beta
        return beta

    def compute_rep_any(self):
        """"""
        left_sign = compute_left_alpha_sign(self.alpha)
        right_alpha = compute_right_alpha(self.alpha, left_sign)

        self.inner_node.alpha = right_alpha
        self.inner_node.compute_leaf_beta()

        beta = np.zeros(self.N, dtype=np.int8)
        beta[:self.inner_node.N] = self.inner_node.beta
        beta[self.inner_node.N:] = self.inner_node.beta
        return beta
