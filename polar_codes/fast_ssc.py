from .base import BasicPolarCode
from .decoders.fast_ssc_decoder import FastSSCDecoder


class FastSSCPolarCode(BasicPolarCode):
    """Polar code with SC decoding algorithm."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder = FastSSCDecoder(
            mask=self.polar_mask,
            is_systematic=self.is_systematic
        )

    def decode(self, received_message):
        """Decode Polar code with Fast SSC decoding algorithm."""
        return self._fast_ssc_decode(received_message)

    def _fast_ssc_decode(self, llr_estimated_message):
        """Fast Simplified Successive cancellation decoder."""
        self.decoder.set_initial_state(llr_estimated_message)
        self.decoder()
        return self._extract(self.decoder.result)
