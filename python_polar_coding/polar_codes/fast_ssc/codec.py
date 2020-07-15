import numpy as np

from python_polar_coding.polar_codes.base import BasePolarCodec

from .decoder import FastSSCDecoder


class FastSSCPolarCodec(BasePolarCodec):
    """Polar code with SC decoding algorithm.

    Based on: https://arxiv.org/pdf/1307.7154.pdf

    """
    decoder_class = FastSSCDecoder

    def init_decoder(self):
        return self.decoder_class(n=self.n, mask=self.mask)

    def decode(self, received_message: np.array) -> np.array:
        """Decode received message presented as LLR values."""
        return self.decoder(received_message)
