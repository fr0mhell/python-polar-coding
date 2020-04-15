from ..base import BasePolarCodec
from .decoder import SCDecoder


class SCPolarCodec(BasePolarCodec):
    """Polar code with SC decoding algorithm."""
    decoder_class = SCDecoder

    def init_decoder(self):
        return self.decoder_class(
            n=self.n, mask=self.mask, is_systematic=self.is_systematic
        )
