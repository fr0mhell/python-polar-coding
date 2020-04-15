import numpy as np
from PyCRC.CRC32 import CRC32
from PyCRC.CRCCCITT import CRCCCITT

from python_polar_coding.polar_codes import utils


class CRC:
    """CRC encoder.

    Supports CRC 32 and CRC 16 CCITT.

    """
    crc_classes = {
        16: CRCCCITT,
        32: CRC32,
    }

    def __init__(self, crc_size):
        self.crc_size = crc_size
        self.crc_coder = self.crc_classes[crc_size]()

    def compute_crc(self, message: np.array) -> np.array:
        """Compute CRC value."""
        return utils.int_to_bin_array(
            value=self._compute_crc(message),
            size=self.crc_size,
        )

    def _compute_crc(self, message: np.array) -> int:
        """Compute CRC bytes value."""
        bit_string = ''.join(str(m) for m in message)
        byte_string = utils.bitstring_to_bytes(bit_string)
        return self.crc_coder.calculate(byte_string)

    def check_crc(self, message: np.array) -> bool:
        """Check if message has errors or not using CRC."""
        received_crc = int(
            ''.join([str(m) for m in message[-self.crc_size::]]),
            2
        )
        check_crc = self._compute_crc(message[:-self.crc_size])
        return received_crc == check_crc
