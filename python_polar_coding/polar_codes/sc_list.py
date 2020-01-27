from typing import Union

from .base.polar_code import BasicPolarCode, BasicPolarCodeWithCRC
from .decoders import SCListDecoder, SCListDecoderWithCRC


class SCListPolarCode(BasicPolarCode):
    """Polar code with SC List  decoding algorithm."""
    decoder_class = SCListDecoder

    def __init__(self, N: int, K: int,
                 design_snr: float = 0.0,
                 is_systematic: bool = True,
                 mask: Union[str, None] = None,
                 pcc_method: str = BasicPolarCode.BHATTACHARYYA,
                 L: int = 1):

        self.L = L
        super().__init__(N=N, K=K,
                         is_systematic=is_systematic,
                         design_snr=design_snr,
                         mask=mask,
                         pcc_method=pcc_method)

    def get_decoder(self):
        return self.decoder_class(n=self.n, mask=self.mask,
                                  is_systematic=self.is_systematic, L=self.L)

    def to_dict(self):
        d = super().to_dict()
        d.update({'L': self.L})
        return d


class SCListPolarCodeWithCRC(BasicPolarCodeWithCRC):
    """Polar code with SC List decoding algorithm and CRC."""
    decoder_class = SCListDecoderWithCRC

    def __init__(self, N: int, K: int,
                 crc_size: int = 32,
                 design_snr: float = 0.0,
                 is_systematic: bool = True,
                 mask: Union[str, None] = None,
                 pcc_method: str = BasicPolarCode.BHATTACHARYYA,
                 L: int = 1):

        self.L = L
        super().__init__(N=N, K=K,
                         is_systematic=is_systematic,
                         design_snr=design_snr,
                         mask=mask,
                         pcc_method=pcc_method,
                         crc_size=crc_size)

    def get_decoder(self):
        return self.decoder_class(n=self.n, mask=self.mask,
                                  is_systematic=self.is_systematic,
                                  L=self.L, crc_codec=self.crc_codec)

    def to_dict(self):
        d = super().to_dict()
        d.update({'L': self.L})
        return d
