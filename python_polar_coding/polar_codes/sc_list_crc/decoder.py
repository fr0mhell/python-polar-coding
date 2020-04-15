import numpy as np

from python_polar_coding.polar_codes.crc import CRC

from ..sc_list import SCListDecoder


class SCListCRCDecoder(SCListDecoder):
    """SC List decoding with CRC."""

    def __init__(self,
                 n: int,
                 mask: np.array,
                 crc_codec: CRC,
                 is_systematic: bool = True,
                 L: int = 1):
        super().__init__(n=n, mask=mask, is_systematic=is_systematic, L=L)
        self.crc_codec = crc_codec

    @property
    def best_result(self):
        """Result from the best path."""
        for result in self.result:
            if self.crc_codec.check_crc(result):
                return result[:-self.crc_codec.crc_size]
        return super().best_result
