import abc

import numpy as np
from anytree import PreOrderIter


class BaseDecoder(metaclass=abc.ABCMeta):
    """Basic class for polar decoder."""

    def __init__(self, n, mask: np.array, is_systematic: bool = True):
        self.N = mask.shape[0]
        self.n = n
        self.is_systematic = is_systematic
        self.mask = mask

    def decode(self, received_llr: np.array) -> np.array:
        decoded = self.decode_internal(received_llr)
        return self.extract_result(decoded)

    @abc.abstractmethod
    def decode_internal(self, received_llr: np.array) -> np.array:
        """Implementation of particular decoding method."""

    def extract_result(self, decoded: np.array) -> np.array:
        """Get decoding result.

        Extract info bits from decoded message due to polar code mask.

        """
        decoded_info = list()

        for i in range(self.N):
            if self.mask[i] == 1:
                decoded_info = np.append(decoded_info, decoded[i])
        return np.array(decoded_info, dtype=np.int)


class BaseTreeDecoder(metaclass=abc.ABCMeta):
    """Basic class for polar decoder that use tree for decoding."""

    node_class: 'BaseDecodingNode'

    def __init__(self, n, mask: np.array):
        self.N = mask.shape[0]
        self.n = n
        self.mask = mask

        self._decoding_tree = self._setup_decoding_tree()
        self._position = 0

    def __call__(self, received_llr: np.array) -> np.array:
        decoded = self.decode(received_llr)
        return self.extract_result(decoded)

    @property
    def leaves(self):
        return self._decoding_tree.leaves

    @property
    def root(self):
        """Returns root node of decoding tree."""
        return self._decoding_tree.root

    @property
    def result(self):
        return self.root.beta

    def decode(self, received_llr: np.array) -> np.array:
        """Implementation of decoding using tree."""
        self._set_initial_state(received_llr)
        self._reset_tree_computed_state()

        for leaf in self.leaves:
            self._set_decoder_state(self._position)
            self._compute_intermediate_alpha(leaf)
            leaf()
            self._compute_intermediate_beta(leaf)
            self._set_next_state(leaf.N)

        return self.result

    def extract_result(self, decoded: np.array) -> np.array:
        """Get decoding result.

        Extract info bits from decoded message due to polar code mask.

        """
        decoded_info = list()

        for i in range(self.N):
            if self.mask[i] == 1:
                decoded_info = np.append(decoded_info, decoded[i])
        return np.array(decoded_info, dtype=np.int)

    def _setup_decoding_tree(self, ):
        """Setup decoding tree."""
        return self.node_class(mask=self.mask)

    def _set_initial_state(self, received_llr):
        """Initialize decoder with received message."""
        self.current_state = np.zeros(self.n, dtype=np.int8)
        self.previous_state = np.ones(self.n, dtype=np.int8)

        # LLR values at intermediate steps
        self._position = 0
        self._decoding_tree.root.alpha = received_llr

    def _reset_tree_computed_state(self):
        """Reset the state of the tree before decoding"""
        for node in PreOrderIter(self._decoding_tree):
            node.is_computed = False

    def _set_decoder_state(self, position):
        """Set current state of the decoder."""
        bits = np.unpackbits(
            np.array([position], dtype=np.uint32).byteswap().view(np.uint8)
        )
        self.current_state = bits[-self.n:]

    @abc.abstractmethod
    def _compute_intermediate_alpha(self, leaf):
        """Compute intermediate Alpha values (LLR)."""

    @abc.abstractmethod
    def _compute_intermediate_beta(self, node):
        """Compute intermediate Beta values (Bits or LLR)."""

    def _set_next_state(self, leaf_size):
        self._position += leaf_size
