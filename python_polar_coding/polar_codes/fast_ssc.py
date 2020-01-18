from .base.polar_code import BasicPolarCode
from .decoders.fast_ssc_decoder import FastSSCDecoder


class FastSSCPolarCode(BasicPolarCode):
    """Polar code with SC decoding algorithm."""
    decoder_class = FastSSCDecoder

    def get_decoder(self):
        return self.decoder_class(n=self.n, mask=self.mask,
                                  is_systematic=self.is_systematic)

    def to_dict(self):
        d = super().to_dict()
        d.update({'M': self.decoder.M})
        return d
