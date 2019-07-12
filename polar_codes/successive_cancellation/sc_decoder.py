import copy

import numpy as np

import utils


class SCDecoder:
    """Implements SC decoding algorithm.

    Can be used for SC List decoding.

    Stores initial and intermediate LLR values, intermediate bit values and
    metrics for forking SC List decoder tree.

    Args:
        received_llr (np.array): LLRs of received message.

    """
    def __init__(self, received_llr):

        self._current_position = 0
        self.current_state = np.zeros(self.n, dtype=np.int8)
        self.previous_step = np.ones(self.n, dtype=np.int8)

        self._msg_length = received_llr.size
        self._steps = int(np.log2(self._msg_length))

        self.received_llr = received_llr
        # LLR values at intermediate steps
        self.intermediate_llr = self._get_intermediate_llr_structure()
        # Bit values at intermediate steps
        self.intermediate_bits = self._get_intermediate_bits_structure()
        # decoded data
        self.decoded = np.zeros(self.N, dtype=np.int8)

        # Value of decoded bit set after forking
        # 0 for LLR > 0, 1 for LLR < 0
        self.fork_value = 1
        # Metric that allow to rate `quality` of the branch - probability of
        # correct decoding decision for the current bit
        self.fork_metric = 1
        # Probability of containing the correct decoded data. Used to evaluate
        # the result after all bits decoded
        self.correct_prob = 1

    @property
    def N(self):
        return self._msg_length

    @property
    def n(self):
        return self._steps

    def decode(self):
        """Decode received message using SC algorithm."""

    def update_llrs(self):
        """Update intermediate LLR values."""

    def update_bits(self):
        """Update intermediate BIT values."""

    def _get_intermediate_llr_structure(self):
        intermediate_llr = []
        length = self.N
        while length > 0:
            length //= 2
            intermediate_llr.append(np.zeros(length, dtype=np.double))
        return intermediate_llr

    def _get_intermediate_bits_structure(self):
        intermediate_bits = []
        length = self.N
        while length > 0:
            length //= 2
            intermediate_bits.append(np.zeros(length, dtype=np.int8))
        return intermediate_bits

    def update_before_fork(self):
        """Update fork parameters.

        LLR = ln(P0) - ln(P1), P0 + P1 = 1
        exp(LLR) = P0 / P1
        P0 = exp(LLR) / (exp(LLR) + 1)
        P1 = 1 / (exp(LLR) + 1)
        Pi = ( (1 - i) * exp(LLR) + i ) / (exp(LLR) + 1)

        """

    def fork(self):
        """Make a copy of SC branch for List decoding"""

    def __eq__(self, other):
        return self.correct_prob == other.correct_prob

    def __gt__(self, other):
        return self.correct_prob > other.correct_prob

    def __ge__(self, other):
        return self > other or self == other

    def __lt__(self, other):
        return not (self >= other)

    def __le__(self, other):
        return not (self > other)

    def update_correct_probability(self):
        """"""

    def update_decoding_position(self, pos):
        """"""

    def decode_current_bit(self):
        """"""

    def set_bit_as_frozen(self):
        """"""


def fork_branches(sc_list, max_list_size):
    """Forks SC branches."""
