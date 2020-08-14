from ..codec import BasePolarCodec
from typing import Optional
import abc


class BaseParallelCoder(BasePolarCodec, metaclass=abc.ABCMeta):
    """"""

    def __init__(
            self,
            N: int,
            K: int,
            sub_codes: int,
            design_snr: Optional[float] = 0.0,
            mask: Optional[str] = None,
            pcc_method: Optional[str] = BasePolarCodec.BHATTACHARYYA,
    ):
        self.sub_codes = sub_codes
        super().__init__(N, K, design_snr, mask, pcc_method)

    def init_encoder(self):
        """Get Polar Encoder instance."""
        return self.encoder_class(
            mask=self.mask,
            n=self.n,
            is_systematic=self.is_systematic,
            sub_codes=self.sub_codes,
        )

    @abc.abstractmethod
    def init_decoder(self):
        """Get Polar Decoder instance."""

    def to_dict(self):
        dct = super().to_dict()
        dct.update({'sub_codes': self.sub_codes})
        return dct
