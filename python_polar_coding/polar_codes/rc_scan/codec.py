from typing import Union

import numpy as np

from python_polar_coding.polar_codes.base import BasePolarCodec

from .decoder import RCSCANDecoder


class RCSCANPolarCodec(BasePolarCodec):
    """Polar code with RC-SCAN decoding algorithm.

    Based on: https://arxiv.org/pdf/1510.06495.pdf,
    DOI: 10.1109/DASIP.2015.7367252

    """
    decoder_class = RCSCANDecoder

    def __init__(
            self,
            N: int,
            K: int,
            design_snr: float = 0.0,
            mask: Union[str, None] = None,
            pcc_method: str = BasePolarCodec.BHATTACHARYYA,
            I: int = 1,
            * args, **kwargs,
    ):
        self.I = I
        super().__init__(N=N, K=K,
                         is_systematic=True,
                         design_snr=design_snr,
                         mask=mask,
                         pcc_method=pcc_method)

    def init_decoder(self):
        return self.decoder_class(n=self.n, mask=self.mask, I=self.I)

    def to_dict(self):
        d = super().to_dict()
        d.update({'I': self.I})
        return d

    def decode(self, received_message: np.array) -> np.array:
        """Decode received message presented as LLR values."""
        return self.decoder(received_message)
