import numba
import numpy as np

from ..base import decoder, functions


class SCDecoder(decoder.BaseDecoder):
    """Implements SC decoding algorithm.

    Stores initial and intermediate LLR values, intermediate bit values and
    metrics for forking SC List decoder tree.

    Args:
        mask (np.array): Polar code mask.
        is_systematic (bool): Systematic code or not

    """

    def __init__(self, n: int, mask: np.array, is_systematic: bool = True):
        super().__init__(n=n, mask=mask, is_systematic=is_systematic)

        self._current_decision = 0

        # LLR values at intermediate steps
        self.intermediate_llr = None
        # Bit values at intermediate steps
        self.intermediate_bits = None
        self.current_state = np.zeros(self.n, dtype=np.int8)
        self.previous_state = np.ones(self.n, dtype=np.int8)

    def decode_internal(self, received_llr: np.array) -> np.array:
        """Implementation of SC decoding method."""
        self._set_initial_state(received_llr)

        for pos in range(self.N):
            self._decode_position(pos)

        return self.result

    @property
    def result(self):
        if self.is_systematic:
            return self.intermediate_bits[0]
        return self.intermediate_bits[-1]

    def _set_initial_state(self, received_llr):
        """Initialize decoder with received message"""
        self.current_state = np.zeros(self.n, dtype=np.int8)
        self.previous_state = np.ones(self.n, dtype=np.int8)
        # LLR values at intermediate steps
        self.intermediate_llr = self._get_intermediate_llr_structure(
            received_llr)
        # Bit values at intermediate steps
        self.intermediate_bits = self._get_intermediate_bits_structure()

    def _get_intermediate_llr_structure(self, received_llr):
        intermediate_llr = [received_llr, ]
        length = self.N // 2
        while length > 0:
            intermediate_llr.append(np.zeros(length, dtype=np.double))
            length //= 2
        return intermediate_llr

    def _get_intermediate_bits_structure(self):
        return [np.zeros(self.N, dtype=np.int8) for _ in range(self.n + 1)]

    def _decode_position(self, position):
        """Decode single position."""
        self._set_decoder_state(position)
        self._compute_intermediate_alpha(position)
        self._compute_beta(position)
        self._compute_intermediate_beta(position)
        self._update_decoder_state()

    def _set_decoder_state(self, position):
        """Set current state of the decoder."""
        bits = np.unpackbits(
            np.array([position], dtype=np.uint32).byteswap().view(np.uint8)
        )
        self.current_state = bits[-self.n:]

    def _compute_intermediate_alpha(self, position):
        """Compute intermediate LLR values."""
        for i in range(1, self.n + 1):
            llr = self.intermediate_llr[i - 1]

            if self.current_state[i - 1] == self.previous_state[i - 1]:
                continue

            if self.current_state[i - 1] == 0:
                self.intermediate_llr[i] = functions.compute_left_alpha(llr)
                continue

            end = position
            start = end - np.power(2, self.n - i)
            left_bits = self.intermediate_bits[i][start: end]
            self.intermediate_llr[i] = functions.compute_right_alpha(llr, left_bits)

    def _compute_beta(self, position):
        """Make decision about current decoding value."""
        mask_bit = self.mask[position]
        self._current_decision = (int(self.intermediate_llr[-1][0] < 0)
                                  if mask_bit == 1 else 0)

    @staticmethod
    @numba.njit
    def _compute_left_alpha(llr):
        """Compute Alpha (LLR) for left node."""
        N = llr.size // 2
        left_llr = np.zeros(N, dtype=np.double)
        for i in range(N):
            left = llr[i]
            right = llr[i + N]
            left_llr[i] = (np.sign(left) * np.sign(right)
                           * np.fabs(np.array([left, right])).min())
        return left_llr

    @staticmethod
    @numba.njit
    def _compute_right_alpha(llr, left_beta):
        """Compute Alpha (LLR) for right node."""
        N = llr.size // 2
        right_llr = np.zeros(N, dtype=np.double)
        for i in range(N):
            right_llr[i] = (llr[i + N] - (2 * left_beta[i] - 1) * llr[i])
        return right_llr

    def _compute_intermediate_beta(self, position):
        """Compute intermediate BIT values."""
        self.intermediate_bits[-1][position] = self._current_decision

        for i in range(self.n - 1, -1, -1):
            source = self.intermediate_bits[i + 1]
            result = self.intermediate_bits[i]

            self.intermediate_bits[i] = functions.compute_encoding_step(
                i, self.n, source, result
            )

    def _update_decoder_state(self):
        """Set next decoding position."""
        self.previous_state.ravel()[:self.n] = self.current_state
