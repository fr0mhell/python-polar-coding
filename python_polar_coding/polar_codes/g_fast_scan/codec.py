from typing import Union

from python_polar_coding.polar_codes.rc_scan import RCSCANPolarCodec

from .decoder import GFastSCANDecoder


class GFastSCANCodec(RCSCANPolarCodec):
    decoder_class = GFastSCANDecoder

    def __init__(
            self,
            N: int,
            K: int,
            design_snr: float = 0.0,
            mask: Union[str, None] = None,
            pcc_method: str = RCSCANPolarCodec.BHATTACHARYYA,
            AF: int = 0,
            I: int = 1,
            * args, **kwargs,
    ):

        self.AF = AF
        super().__init__(
            N=N,
            K=K,
            design_snr=design_snr,
            mask=mask,
            pcc_method=pcc_method,
            I=I,
        )

    def init_decoder(self):
        return self.decoder_class(
            n=self.n,
            mask=self.mask,
            AF=self.AF,
            I=self.I,
        )

    def to_dict(self):
        d = super().to_dict()
        d.update({'AF': self.AF})
        return d
