import numpy as np
from numba import njit

from .crc import CRC


class Encoder:
    """Polar Codes encoder.

    """

    def __init__(self,
                 mask: np.array,
                 n: int,
                 is_systematic: bool = True):

        self.n = n
        self.N = mask.shape[0]
        self.mask = mask
        self.is_systematic = is_systematic

    def encode(self, message: np.array) -> np.array:
        """"""
        precoded = self._precode(message)
        encoded = self._non_systematic_encode(precoded)

        if self.is_systematic:
            encoded *= self.mask
            encoded = self._non_systematic_encode(encoded)

        return encoded

    def _precode(self, message: np.array) -> np.array:
        """Apply polar code mask to information message.

        Replace 1's of polar code mask with bits of information message.

        """
        precoded = np.zeros(self.N, dtype=int)
        precoded[self.mask == 1] = message
        return precoded

    @staticmethod
    @njit
    def _non_systematic_encode(message: np.array, n: int) -> np.array:
        """Non-systematic encoding.

        Args:
            message (numpy.array): precoded message to encode.

        Returns:
            message (numpy.array): non-systematically encoded message.

        """
        for i in range(n - 1, -1, -1):
            pairs_per_group = step = np.power(2, n - i - 1)
            groups = np.power(2, i)

            for g in range(groups):
                start = 2 * g * step

                for p in range(pairs_per_group):
                    message[p + start] = message[p + start] ^ message[p + start + step]
                    message[p + start + step] = message[p + start + step]

        return message


class EncoderWithCRC(Encoder):
    """Polar Encoder with CRC support."""

    def __init__(self,
                 mask: np.array,
                 n: int,
                 crc_codec: CRC,
                 is_systematic: bool = True,):
        super().__init__(mask, n, is_systematic)
        self.crc_codec = crc_codec

    def encode(self, message: np.array):
        """Compute CRC value and append to message before encoding."""
        message = np.append(
            message,
            self.crc_codec.compute_crc(message),
        )
        return super().encode(message)
