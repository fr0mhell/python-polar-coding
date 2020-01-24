from .base.polar_code import BasicPolarCode
from .decoders import SCDecoder


class SCPolarCode(BasicPolarCode):
    """Polar code with SC decoding algorithm."""
    decoder_class = SCDecoder

    def get_decoder(self):
        return self.decoder_class(
            n=self.n, mask=self.mask, is_systematic=self.is_systematic
        )
