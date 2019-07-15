import copy

import numpy as np

import utils

from .functions import compute_left_llr, compute_right_llr


class SCDecoder:
    """Implements SC decoding algorithm.

    Can be used for SC List decoding.

    Stores initial and intermediate LLR values, intermediate bit values and
    metrics for forking SC List decoder tree.

    Args:
        received_llr (np.array): LLRs of received message.

    """
    def __init__(self, received_llr):

        self._msg_length = received_llr.size
        self._steps = int(np.log2(self._msg_length))
        self._state_dimension = int(np.ceil(np.log2(self.n)))

        self._current_position = 0
        self._current_state = np.zeros(self.n, dtype=np.int8)
        self._previous_state = np.ones(self.n, dtype=np.int8)
        self._current_level = 0

        self.received_llr = received_llr
        # LLR values at intermediate steps
        self.intermediate_llr = self._get_intermediate_llr_structure()
        # Bit values at intermediate steps
        self.intermediate_left_bits = self._get_intermediate_bits_structure()
        self.intermediate_right_bits = self._get_intermediate_bits_structure()
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
        self.set_decoder_state()
        self.compute_intermediate_llr()
        self.make_decision()
        self.compute_intermediate_bits()
        self.set_next_decoding_position()

    def set_decoder_state(self):
        """Set current state of the decoder."""
        bits = np.unpackbits(np.array([self._current_position], dtype=np.uint8))
        self._current_state = bits[-self.n:]
        self._current_level = int(np.argwhere(self._current_state != self._previous_state)[0])

    def compute_intermediate_llr(self):
        """Compute intermediate LLR values."""
        llr = (self.received_llr if self._current_level == 0
               else self.intermediate_llr[self._current_level])

        for index, value in enumerate(self._current_state[self._current_level:]):
            self.intermediate_llr[index] = (
                compute_left_llr(llr) if value == 0
                else compute_right_llr(llr, self.intermediate_left_bits[index + 1])
            )

            llr = self.intermediate_llr[index]

    def make_decision(self, mask_bit):
        """Make decision about current decoding value."""
        return 0 if mask_bit == 0 else int(self.intermediate_llr[-1][0] < 0)

    def compute_intermediate_bits(self):
        """Compute intermediate BIT values."""

    def set_next_decoding_position(self):
        """Set next decoding position."""
        self._current_position += 1

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
