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

        self.current_position = 0
        self.reversed_position = 0
        self.current_state = np.zeros(self.n, dtype=np.int8)
        self.current_level = 0
        self.previous_state = np.ones(self.n, dtype=np.int8)
        self.previous_level = 0

        self.is_systematic = is_systematic
        self.received_llr = received_llr
        self.mask = mask
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

    def decoder_step(self, step):
        """Single step of SC-decoding algorithm to decode one bit."""
        self.set_decoder_state(step)
        self.compute_intermediate_llr()
        decoded = self.make_decision()

        self.compute_intermediate_bits(decoded)
        self.set_next_decoding_position()

    def set_decoder_state(self, step):
        """Set current state of the decoder."""
        self.current_position = step
        self.reversed_position = bitreversed(self.current_position, self.n)
        bits = np.unpackbits(np.array(
            [self.current_position], dtype=np.uint32
        ).byteswap().view(np.uint8))
        self.current_state = bits[-self.n:]
        self.current_level = \
            np.argwhere(self.current_state != self.previous_state)[0][0]

    def compute_intermediate_llr(self):
        """Compute intermediate LLR values."""
        llr = (self.received_llr if self.current_level == 0
               else self.intermediate_llr[self.current_level - 1])

        for index, value in enumerate(self.current_state):
            if self.current_state[index] == self.previous_state[index]:
                continue

            if value == 0:
                self.intermediate_llr[index] = self.compute_left_llr(llr)

            if value == 1:
                end = self.N
                start = self.reversed_position - np.power(2, index)
                step = np.power(2, index + 1)
                bits = self.intermediate_bits[index + 1][start:end:step]
                self.intermediate_llr[index] = self.compute_right_llr(llr, bits)

            llr = self.intermediate_llr[index]

    def make_decision(self):
        """Make decision about current decoding value."""
        mask_bit = self.mask[self.reversed_position]
        result = 0 if mask_bit == 0 else int(self.intermediate_llr[-1][0] < 0)
        return result

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

    def set_next_decoding_position(self):
        """Set next decoding position."""
        self.previous_state.ravel()[:self.n] = self.current_state
        self.previous_level = self.current_level

    def _get_intermediate_llr_structure(self):
        intermediate_llr = list()
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
