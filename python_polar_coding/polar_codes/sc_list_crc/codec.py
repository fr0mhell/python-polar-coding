from typing import Union

from ..base.codec import BaseCRCPolarCodec
from .decoder import SCListCRCDecoder


class SCListCRCPolarCodec(BaseCRCPolarCodec):
    """Polar code with SC List decoding algorithm and CRC."""
    decoder_class = SCListCRCDecoder

    def __init__(self, N: int, K: int,
                 crc_size: int = 32,
                 design_snr: float = 0.0,
                 is_systematic: bool = True,
                 mask: Union[str, None] = None,
                 pcc_method: str = BaseCRCPolarCodec.BHATTACHARYYA,
                 L: int = 1):

        self.L = L
        super().__init__(N=N, K=K,
                         is_systematic=is_systematic,
                         design_snr=design_snr,
                         mask=mask,
                         pcc_method=pcc_method,
                         crc_size=crc_size)

    def init_decoder(self):
        return self.decoder_class(n=self.n, mask=self.mask,
                                  is_systematic=self.is_systematic,
                                  L=self.L, crc_codec=self.crc_codec)

    def to_dict(self):
        d = super().to_dict()
        d.update({'L': self.L})
        return d
