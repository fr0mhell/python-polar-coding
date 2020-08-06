from typing import Union

from python_polar_coding.polar_codes.fast_ssc import FastSSCPolarCodec

from .decoder import GFastSSCDecoder


class GFastSSCPolarCodec(FastSSCPolarCodec):
    """Generalized Fast SSC code.

    Based on: https://arxiv.org/pdf/1804.09508.pdf

    """
    decoder_class = GFastSSCDecoder

    def __init__(
            self,
            N: int,
            K: int,
            design_snr: float = 0.0,
            mask: Union[str, None] = None,
            pcc_method: str = FastSSCPolarCodec.BHATTACHARYYA,
            AF: int = 0,
    ):

        self.AF = AF
        super().__init__(
            N=N,
            K=K,
            design_snr=design_snr,
            mask=mask,
            pcc_method=pcc_method,
        )

    def init_decoder(self):
        return self.decoder_class(n=self.n, mask=self.mask, AF=self.AF)

    def to_dict(self):
        d = super().to_dict()
        d.update({'AF': self.AF})
        return d
