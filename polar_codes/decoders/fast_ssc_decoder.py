import numba
import numpy as np
from anytree import Node

from . import SCDecoder


class FastSSCNode(Node):

    LEFT = 'left_child'
    RIGHT = 'right_child'
    ROOT = 'root'
    NODE_NAMES = (LEFT, RIGHT, ROOT)

    ZERO_NODE = 'ZERO'
    ONE_NODE = 'ONE'
    SINGLE_PARITY_CHECK = 'SINGLE_PARITY_CHECK'
    REPETITION = 'REPETITION'
    OTHER = 'OTHER'

    FAST_SSC_NODE_TYPES = (ZERO_NODE, ONE_NODE, SINGLE_PARITY_CHECK, REPETITION)  # noqa

    # Minimal size of Single parity check node
    SPC_MIN_SIZE = 4
    # Minimal size of Repetition Fast SSC Node
    REPETITION_MIN_SIZE = 2

    def __init__(self, mask, name=ROOT, **kwargs):
        """A node of Fast SSC decoder."""
        if name not in FastSSCNode.NODE_NAMES:
            raise ValueError('Wrong Fast SSC Node type')

        super().__init__(name, **kwargs)

        self._mask = mask
        self._node_type = self._get_node_type()
        self._llr = np.zeros(self.N, dtype=np.double)
        self._bits = np.zeros(self.N, dtype=np.int8)

        self._build_fast_ssc_tree()

    @property
    def N(self):
        return self._mask.size

    @property
    def llr(self):
        return self._llr

    @llr.setter
    def llr(self, value):
        if self._mask.size != value.size:
            raise ValueError('Wrong size of LLR vector')
        self._llr = np.array(value)

    @property
    def bits(self):
        return self._bits

    @bits.setter
    def bits(self, value):
        if self._mask.size != value.size:
            raise ValueError('Wrong size of Bits vector')
        self._bits = np.array(value)

    @property
    def is_left(self):
        return self._node_type == FastSSCNode.LEFT

    @property
    def is_right(self):
        return self._node_type == FastSSCNode.RIGHT

    @property
    def is_fast_ssc_node(self):
        return self._node_type in FastSSCNode.FAST_SSC_NODE_TYPES

    def make_decision(self):
        if not self.is_leaf:
            raise TypeError('Cannot make decision in not a leaf node.')

        if self._node_type == FastSSCNode.ZERO_NODE:
            self._bits = np.zeros(self.N)
        if self._node_type == FastSSCNode.ONE_NODE:
            self._bits = self._make_hard_decision(self.llr)
        if self._node_type == FastSSCNode.SINGLE_PARITY_CHECK:
            self._bits = self._compute_bits_spc(self.llr)
        if self._node_type == FastSSCNode.REPETITION:
            self._bits = self._compute_bits_repetition(self.llr)

    @staticmethod
    # @numba.njit
    def _make_hard_decision(llr):
        # njit incompatible operation
        return np.array(llr < 0, dtype=np.int8)

    @staticmethod
    # @numba.njit
    def _compute_bits_spc(llr):
        # njit incompatible operation
        bits = np.array(llr < 0, dtype=np.int8)
        parity = np.sum(bits) % 2
        arg_min = np.abs(llr).argmin()
        bits[arg_min] = (bits[arg_min] + parity) % 2
        return bits

    @staticmethod
    # @numba.njit
    def _compute_bits_repetition(llr):
        return np.zeros(llr.size) if np.sum(llr) >= 0 else np.ones(llr.size)

    def _get_node_type(self):
        """Get the type of Fast SSC Node.

        * Zero node - [0, 0, 0, 0, 0, 0, 0, 0];
        * One node - [1, 1, 1, 1, 1, 1, 1, 1];
        * Single parity check node - [0, 1, 1, 1, 1, 1, 1, 1];
        * Repetition node - [0, 0, 0, 0, 0, 0, 0, 1].

        Or other type.

        """
        if np.all(self._mask == 0):
            return FastSSCNode.ZERO_NODE
        if np.all(self._mask == 1):
            return FastSSCNode.ONE_NODE
        if (self._mask.size >= FastSSCNode.SPC_MIN_SIZE
                and self._mask[0] == 0 and np.sum(self._mask) == self.N - 1):
            return FastSSCNode.SINGLE_PARITY_CHECK
        if (self._mask.size >= FastSSCNode.REPETITION_MIN_SIZE
                and self._mask[-1] == 1 and np.sum(self._mask) == 1):
            return FastSSCNode.REPETITION
        return FastSSCNode.OTHER

    def _build_fast_ssc_tree(self):
        """Build Fast SSC tree."""
        if self.is_fast_ssc_node:
            return

        left_mask, right_mask = np.split(self._mask, 2)
        FastSSCNode(mask=left_mask, name=FastSSCNode.LEFT, parent=self)
        FastSSCNode(mask=right_mask, name=FastSSCNode.RIGHT, parent=self)
