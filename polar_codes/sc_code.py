from .base import BasicPolarCode
from .successive_cancellation import SCDecoder


class SCPolarCode(BasicPolarCode):
    """Polar code with SC decoding algorithm."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.decoder = None

    def decode(self, received_message):
        """Decode Polar code with SC decoding algorithm."""
        return self._sc_decode(received_message)

    def _sc_decode(self, llr_estimated_message):
        """Successive cancellation decoder

        Based on: https://arxiv.org/abs/0807.3917 (page 15).

        """
        self.decoder = SCDecoder(
            received_llr=llr_estimated_message,
            mask=self.polar_mask,
            is_systematic=self.is_systematic
        )

        for step in range(self.N):
            self.decoder.decoder_step(step)

        # if self.is_systematic:
        #     # for systematic code first need to mul decoding result with
        #     # code generator matrix, and then extract information bits due to
        #     # polar coding matrix
        #     return self._extract(self._mul_matrix(self.decoder.result))
        return self._extract(self.decoder.result)
