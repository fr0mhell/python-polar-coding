import numba
import numpy as np

from ..base.functions import compute_encoding_step


class SCDecoder:
    """Implements SC decoding algorithm.

    Can be used for SC List decoding.

    Stores initial and intermediate LLR values, intermediate bit values and
    metrics for forking SC List decoder tree.

    Args:
        received_llr (np.array): LLRs of received message.
        mask (np.array): Polar code mask.

    """
    def __init__(self, received_llr, mask, is_systematic=True):

        self._msg_length = received_llr.size
        self._steps = int(np.log2(self._msg_length))
        self._state_dimension = int(np.ceil(np.log2(self.n)))

        self.current_state = np.zeros(self.n, dtype=np.int8)
        self.previous_state = np.ones(self.n, dtype=np.int8)

        self.is_systematic = is_systematic
        self.mask = mask
        # LLR values at intermediate steps
        self.intermediate_llr = self._get_intermediate_llr_structure(
            received_llr)
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

    @property
    def N(self):
        return self._msg_length

    @property
    def n(self):
        return self._steps

    @property
    def result(self):
        """Decoding result."""
        if self.is_systematic:
            return self.intermediate_bits[0]
        return self.intermediate_bits[-1]

    def decode_position(self, position):
        """Single step of SC-decoding algorithm to decode one bit."""
        self.set_decoder_state(position)
        self.compute_intermediate_llr(position)
        result = self.make_decision(position)

        self.compute_intermediate_bits(result, position)
        self.update_decoder_state()

    def set_decoder_state(self, position):
        """Set current state of the decoder."""
        bits = np.unpackbits(
            np.array(
                [position], dtype=np.uint32).byteswap().view(np.uint8)
        )
        self.current_state = bits[-self.n:]

    def compute_intermediate_llr(self, position):
        """Compute intermediate LLR values."""
        for i in range(1, self.n + 1):
            llr = self.intermediate_llr[i - 1]

            if self.current_state[i - 1] == self.previous_state[i - 1]:
                continue

            if self.current_state[i - 1] == 0:
                self.intermediate_llr[i] = self.compute_left_llr(llr)
                continue

            end = position
            start = end - np.power(2, self.n - i)
            left_bits = self.intermediate_bits[i][start: end]
            self.intermediate_llr[i] = self.compute_right_llr(llr, left_bits)

    def make_decision(self, position):
        """Make decision about current decoding value."""
        mask_bit = self.mask[position]
        return int(self.intermediate_llr[-1][0] < 0) if mask_bit else 0

    def compute_intermediate_bits(self, decoded, position):
        """Compute intermediate BIT values."""
        self.intermediate_bits[-1][position] = decoded

        for i in range(self.n - 1, -1, -1):
            if self.current_state[i] == self.previous_state[i]:
                continue

            source = self.intermediate_bits[i + 1]
            result = self.intermediate_bits[i]

            self.intermediate_bits[i] = compute_encoding_step(
                i, self.n, source, result
            )

    def update_decoder_state(self):
        """Set next decoding position."""
        self.previous_state.ravel()[:self.n] = self.current_state

    def _get_intermediate_llr_structure(self, received_llr):
        intermediate_llr = [received_llr, ]
        length = self.N // 2
        while length > 0:
            intermediate_llr.append(np.zeros(length, dtype=np.double))
            length //= 2
        return intermediate_llr

    def _get_intermediate_bits_structure(self):
        intermediate_bits = [
            np.zeros(self.N, dtype=np.int8) for _ in range(self.n + 1)
        ]
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

    def update_correct_probability(self):
        """"""

    def update_decoding_position(self, pos):
        """"""

    def decode_current_bit(self):
        """"""

    def set_bit_as_frozen(self):
        """"""

    @staticmethod
    @numba.njit
    def compute_left_llr(llr):
        """Compute LLR for left node."""
        N = llr.size // 2
        left_llr = np.zeros(N, dtype=np.double)
        for i in range(N):
            left = llr[i]
            right = llr[i + N]
            left_llr[i] = (
                np.sign(left) * np.sign(right) *
                np.fabs(np.array([left, right])).min()
            )
        return left_llr

    @staticmethod
    @numba.njit
    def compute_right_llr(llr, left_bits):
        """Compute LLR for right node."""
        N = llr.size // 2
        right_llr = np.zeros(N, dtype=np.double)
        for i in range(N):
            right_llr[i] = (llr[i + N] - (2 * left_bits[i] - 1) * llr[i])
        return right_llr


def fork_branches(sc_list, max_list_size):
    """Forks SC branches."""
