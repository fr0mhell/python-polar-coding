from .base import BasicPolarCode
from .decoders import SCDecoder


class SCPolarCode(BasicPolarCode):
    """Polar code with SC decoding algorithm."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder = SCDecoder(
            mask=self.polar_mask,
            is_systematic=self.is_systematic
        )

    def decode(self, received_message):
        """Decode Polar code with SC decoding algorithm."""
        return self._sc_decode(received_message)

    def _sc_decode(self, llr_estimated_message):
        """Successive cancellation decoder

        Based on: https://arxiv.org/abs/0807.3917 (page 15).

        """
        self.decoder.set_initial_state(llr_estimated_message)

        for step in range(self.N):
            self.decoder(step)
        return self._extract(self.decoder.result)
