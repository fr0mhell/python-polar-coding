import numpy as np
from anytree import Node

from python_polar_coding.polar_codes.base import make_hard_decision

from .functions import compute_repetition, compute_single_parity_check


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

    # Minimal size of Single parity check node
    SPC_MIN_SIZE = 4
    # Minimal size of Repetition Fast SSC Node
    REPETITION_MIN_SIZE = 2

    def __init__(self, mask, name=ROOT, N_min=None, **kwargs):
        """A node of Fast SSC decoder."""
        if name not in self.__class__.NODE_NAMES:
            raise ValueError('Wrong Fast SSC Node type')

        super().__init__(name, **kwargs)

        self._mask = mask
        self.N_min = N_min
        self._node_type = self.get_node_type()
        self._alpha = np.zeros(self.N, dtype=np.double)
        self._beta = np.zeros(self.N, dtype=np.int8)

        self.is_computed = False
        self._build_decoding_tree()

    def __str__(self):
        return ''.join([str(m) for m in self._mask])

    @property
    def N(self):
        return self._mask.size

    @property
    def M(self):
        """Minimal size of component polar code."""
        return self.N_min

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if self._mask.size != value.size:
            raise ValueError('Wrong size of LLR vector')
        self._alpha = np.array(value)

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        if self._mask.size != value.size:
            raise ValueError('Wrong size of Bits vector')
        self._beta = np.array(value)

    @property
    def is_left(self):
        return self.name == self.__class__.LEFT

    @property
    def is_right(self):
        return self.name == self.__class__.RIGHT

    @property
    def is_simplified_node(self):
        return self._node_type != self.__class__.OTHER

    @property
    def zero_min_size(self):
        return self.M or 1

    @property
    def one_min_size(self):
        return self.zero_min_size

    @property
    def repetition_min_size(self):
        return self.M or self.__class__.REPETITION_MIN_SIZE

    @property
    def spc_min_size(self):
        return self.M or self.__class__.SPC_MIN_SIZE

    def to_dict(self):
        return {
            'type': self._node_type,
            'mask': self._mask,
        }

    def compute_leaf_beta(self):
        if not self.is_leaf:
            raise TypeError('Cannot make decision in not a leaf node.')

        if self._node_type == FastSSCNode.ZERO_NODE:
            self._beta = np.zeros(self.N, dtype=np.int8)
        if self._node_type == FastSSCNode.ONE_NODE:
            self._beta = make_hard_decision(self.alpha)
        if self._node_type == FastSSCNode.SINGLE_PARITY_CHECK:
            self._beta = compute_single_parity_check(self.alpha)
        if self._node_type == FastSSCNode.REPETITION:
            self._beta = compute_repetition(self.alpha)

    def _initialize_beta(self):
        """Initialize BETA values on tree building."""
        return np.zeros(self.N, dtype=np.int8)

    def get_node_type(self):
        """Get the type of Fast SSC Node.

        * Zero node - [0, 0, 0, 0, 0, 0, 0, 0];
        * One node - [1, 1, 1, 1, 1, 1, 1, 1];
        * Single parity check node - [0, 1, 1, 1, 1, 1, 1, 1];
        * Repetition node - [0, 0, 0, 0, 0, 0, 0, 1].

        Or other type.

        """
        if self._check_is_zero(self._mask) and self.N >= self.zero_min_size:
            return FastSSCNode.ZERO_NODE
        if self._check_is_one(self._mask) and self.N >= self.one_min_size:
            return FastSSCNode.ONE_NODE
        if self.N >= self.repetition_min_size and self._check_is_rep(self._mask):  # noqa
            return FastSSCNode.REPETITION
        if self.N >= self.spc_min_size and self._check_is_spc(self._mask):
            return FastSSCNode.SINGLE_PARITY_CHECK
        return FastSSCNode.OTHER

    def _check_is_one(self, mask):
        return np.all(mask == 1)

    def _check_is_zero(self, mask):
        return np.all(mask == 0)

    def _check_is_spc(self, mask):
        return mask[0] == 0 and np.sum(mask) == mask.size - 1

    def _check_is_rep(self, mask):
        return mask[-1] == 1 and np.sum(mask) == 1

    def _build_decoding_tree(self):
        """Build Fast SSC decoding tree."""
        if self.is_simplified_node:
            return

        if self._mask.size == self.M:
            return

        left_mask, right_mask = np.split(self._mask, 2)
        cls = self.__class__
        cls(mask=left_mask, name=self.LEFT, N_min=self.M, parent=self)
        cls(mask=right_mask, name=self.RIGHT, N_min=self.M, parent=self)
