import abc

import numpy as np


class BaseDecoder(metaclass=abc.ABCMeta):
    """Basic class for polar decoder."""

    def __init__(self, n, mask: np.array, is_systematic: bool = True):
        self.N = mask.shape[0]
        self.n = n
        self.is_systematic = is_systematic
        self.mask = mask

    def decode(self, received_llr: np.array) -> np.array:
        decoded = self.decode_internal(received_llr)
        return self.get_result(decoded)

    @abc.abstractmethod
    def decode_internal(self, received_llr: np.array) -> np.array:
        """Implementation of particular decoding method."""

    def get_result(self, decoded: np.array) -> np.array:
        """Get decoding result.

        Extract info bits from decoded message due to polar code mask.

        """
        decoded_info = list()

        for i in range(self.N):
            if self.mask[i] == 1:
                decoded_info = np.append(decoded_info, decoded[i])
        return np.array(decoded_info, dtype=np.int)
