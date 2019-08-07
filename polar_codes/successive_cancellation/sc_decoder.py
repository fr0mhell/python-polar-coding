import numba
import numpy as np

from utils import bitreversed


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

        self.compute_intermediate_bits(result)
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

            if self.current_state[i] == self.previous_state[i]:
                continue

            if self.current_state[i] == 0:
                self.intermediate_llr[i] = self.compute_left_llr(llr)
                continue

            end = position
            start = end - np.array(2, self.n - i)
            left_bits = self.intermediate_bits[i][start: end]
            self.intermediate_bits[i] = self.compute_right_llr(llr, left_bits)

    def make_decision(self, position):
        """Make decision about current decoding value."""
        mask_bit = self.mask[position]
        return int(self.intermediate_llr[-1][0] < 0) if mask_bit == 0 else 0

    def compute_intermediate_bits(self, decoded):
        """Compute intermediate BIT values."""
        self.intermediate_bits[-1][self.reversed_position] = decoded

        if self.current_position % 2 == 0:
            return

        msx_level = self._compute_max_level(self.current_position + 1, self.n)
        for i in range(self.n, msx_level, -1):
            bits = self.intermediate_bits[i]
            self.intermediate_bits[i - 1] = self.compute_bits(bits, i, self.n)

    @staticmethod
    @numba.njit
    def _compute_max_level(end, n):
        """"""
        x = np.log2(end)
        if x == n:
            return 0
        if x != int(x):
            return n - 1
        return n - int(x)

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
        left_llr = np.zeros(llr.size // 2, dtype=np.double)
        for i in range(left_llr.size):
            left_llr[i] = (
                np.sign(llr[2 * i]) *
                np.sign(llr[2 * i + 1]) *
                np.fabs(llr[2 * i: 2 * i + 2]).min()
            )
        return left_llr

    @staticmethod
    @numba.njit
    def compute_right_llr(llr, left_bits):
        """Compute LLR for right node."""
        right_llr = np.zeros(llr.size // 2, dtype=np.double)
        for i in range(right_llr.size):
            right_llr[i] = (
                llr[2 * i + 1] -
                (2 * left_bits[i] - 1) * llr[2 * i]
            )
        return right_llr

    @staticmethod
    @numba.njit
    def compute_bits(bits, current_level, n):
        """Compute intermediate bits."""
        result = np.copy(bits)

        step = np.power(2, current_level - 1)
        pairs = np.power(2, n - current_level)

        for i in range(pairs):
            start = 2 * i * step

            for j in range(step):
                result[j+start] = result[j+start] ^ result[j+start+step]

        return result


def fork_branches(sc_list, max_list_size):
    """Forks SC branches."""
