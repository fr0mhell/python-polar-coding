import numba
import numpy as np

from ..base.functions import basic_llr_computation, compute_encoding_step


class SCDecoder:
    """Implements SC decoding algorithm.

    Stores initial and intermediate LLR values, intermediate bit values and
    metrics for forking SC List decoder tree.

    Args:
        mask (np.array): Polar code mask.
        is_systematic (bool): Systematic code or not

    """
    def __init__(self, mask, is_systematic=True):

        self._msg_length = mask.size
        self._steps = int(np.log2(self._msg_length))
        self.is_systematic = is_systematic
        self.mask = mask

        self._current_decision = 0

        # LLR values at intermediate steps
        self.intermediate_llr = None
        # Bit values at intermediate steps
        self.intermediate_bits = None
        self.current_state = np.zeros(self.n, dtype=np.int8)
        self.previous_state = np.ones(self.n, dtype=np.int8)

    def set_initial_state(self, received_llr):
        """Initialize decoder with received message"""
        self.current_state = np.zeros(self.n, dtype=np.int8)
        self.previous_state = np.ones(self.n, dtype=np.int8)
        # LLR values at intermediate steps
        self.intermediate_llr = self._get_intermediate_llr_structure(
            received_llr)
        # Bit values at intermediate steps
        self.intermediate_bits = self._get_intermediate_bits_structure()

    def __call__(self, position, *args, **kwargs):
        """Single step of SC-decoding algorithm to decode one bit."""
        self.set_decoder_state(position)
        self.compute_intermediate_alpha(position)
        self.compute_beta(position)
        self.compute_intermediate_beta(position)
        self.update_decoder_state()

    @property
    def result(self):
        """Decoding result."""
        if self.is_systematic:
            return self.intermediate_bits[0]
        return self.intermediate_bits[-1]

    @property
    def N(self):
        return self._msg_length

    @property
    def n(self):
        return self._steps

    def set_decoder_state(self, position):
        """Set current state of the decoder."""
        bits = np.unpackbits(
            np.array([position], dtype=np.uint32).byteswap().view(np.uint8)
        )
        self.current_state = bits[-self.n:]

    def compute_intermediate_alpha(self, position):
        """Compute intermediate LLR values."""
        for i in range(1, self.n + 1):
            llr = self.intermediate_llr[i - 1]

            if self.current_state[i - 1] == self.previous_state[i - 1]:
                continue

            if self.current_state[i - 1] == 0:
                self.intermediate_llr[i] = self.compute_left_alpha(llr)
                continue

            end = position
            start = end - np.power(2, self.n - i)
            left_bits = self.intermediate_bits[i][start: end]
            self.intermediate_llr[i] = self.compute_right_alpha(llr, left_bits)

    def compute_beta(self, position):
        """Make decision about current decoding value."""
        mask_bit = self.mask[position]
        self._current_decision = (
            int(self.intermediate_llr[-1][0] < 0) if mask_bit == 1 else 0
        )

    def compute_intermediate_beta(self, position):
        """Compute intermediate BIT values."""
        self.intermediate_bits[-1][position] = self._current_decision

        for i in range(self.n - 1, -1, -1):
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
        intermediate_bits = [np.zeros(self.N, dtype=np.int8)
                             for _ in range(self.n + 1)]
        return intermediate_bits

    @staticmethod
    @numba.njit
    def compute_left_alpha(llr):
        """Compute Alpha (LLR) for left node."""
        N = llr.size // 2
        left_llr = np.zeros(N, dtype=np.double)
        for i in range(N):
            left = llr[i]
            right = llr[i + N]
            left_llr[i] = basic_llr_computation(left, right)
        return left_llr

    @staticmethod
    @numba.njit
    def compute_right_alpha(llr, left_beta):
        """Compute Alpha (LLR) for right node."""
        N = llr.size // 2
        right_llr = np.zeros(N, dtype=np.double)
        for i in range(N):
            right_llr[i] = (llr[i + N] - (2 * left_beta[i] - 1) * llr[i])
        return right_llr
