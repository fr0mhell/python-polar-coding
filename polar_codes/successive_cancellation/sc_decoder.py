import numpy as np

from .functions import compute_left_llr, compute_right_llr, compute_bits


class SCDecoder:
    """Implements SC decoding algorithm.

    Can be used for SC List decoding.

    Stores initial and intermediate LLR values, intermediate bit values and
    metrics for forking SC List decoder tree.

    Args:
        received_llr (np.array): LLRs of received message.
        mask (np.array): Polar code mask.

    """
    def __init__(self, received_llr, mask):

        self._msg_length = received_llr.size
        self._steps = int(np.log2(self._msg_length))
        self._state_dimension = int(np.ceil(np.log2(self.n)))

        self.current_position = 0
        self.current_state = np.zeros(self.n, dtype=np.int8)
        self.previous_state = np.ones(self.n, dtype=np.int8)
        self.current_level = 0

        self.received_llr = received_llr
        self.mask = mask
        # decoded data
        self.decoded = np.zeros(self.N, dtype=np.int8)
        # LLR values at intermediate steps
        self.intermediate_llr = self._get_intermediate_llr_structure()
        # Bit values at intermediate steps
        self.intermediate_bits = self._get_intermediate_bits_structure()

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

    def decoder_step(self):
        """Single step of SC-decoding algorithm to decode one bit."""
        self.set_decoder_state()
        self.compute_intermediate_llr()
        self.make_decision()

        if self.current_position == self.N - 1:
            return

        self.compute_intermediate_bits()
        self.set_next_decoding_position()

    def set_decoder_state(self):
        """Set current state of the decoder."""
        bits = np.unpackbits(np.array([self.current_position], dtype=np.uint8))
        self.current_state = bits[-self.n:]
        self.current_level = np.argwhere(self.current_state != self.previous_state)[0][0]

    def compute_intermediate_llr(self):
        """Compute intermediate LLR values."""
        llr = (self.received_llr if self.current_level == 0
               else self.intermediate_llr[self.current_level - 1])

        for index, value in enumerate(self.current_state):
            if self.current_state[index] == self.previous_state[index]:
                continue

            if value == 0:
                self.intermediate_llr[index] = compute_left_llr(llr)

            if value == 1:
                end = self.current_position
                if self.current_position % 2 == 1:
                    start = self.current_position - 1
                else:
                    start = self.current_position - int(np.power(2, self.current_level))

                bits = self.intermediate_bits[index][start:end]
                self.intermediate_llr[index] = compute_right_llr(llr, bits)

            llr = self.intermediate_llr[index]

    def make_decision(self):
        """Make decision about current decoding value."""
        mask_bit = self.mask[self.current_position]
        result = 0 if mask_bit == 0 else int(self.intermediate_llr[-1][0] < 0)
        self.decoded[self.current_position] = result

    def compute_intermediate_bits(self):
        """Compute intermediate BIT values."""
        self.intermediate_bits[-1][self.current_position] = \
            self.decoded[self.current_position]

        if self.current_position % 2 == 0:
            return

        max_level = self.n - np.argwhere(self.current_state != 0)[0][-1] + 1
        for i in range(1, max_level):
            start = self.current_position - i
            end = int(np.power(2, i))
            middle = end // 2

            left_bits = self.intermediate_bits[self.n - i][start:middle]
            right_bits = self.intermediate_bits[self.n - i][middle:end]
            self.intermediate_bits[i][start:end] = compute_bits(left_bits, right_bits)

    def set_next_decoding_position(self):
        """Set next decoding position."""
        self.current_position += 1
        self.previous_state.ravel()[:self.n] = self.current_state

    def _get_intermediate_llr_structure(self):
        intermediate_llr = list()
        length = self.N // 2
        while length > 0:
            intermediate_llr.append(np.zeros(length, dtype=np.double))
            length //= 2
        return intermediate_llr

    def _get_intermediate_bits_structure(self):
        intermediate_bits = list()
        for _ in range(self.n):
            # Set initial values to -1 for debugging and testing purposes
            intermediate_bits.append(np.zeros(self.N, dtype=np.int8) - 1)
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
