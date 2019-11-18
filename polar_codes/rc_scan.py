from .base import BasicPolarCode
from .decoders.rc_scan_decoder import RCSCANDecoder


class RCSCANPolarCode(BasicPolarCode):
    """Polar code with RC-SCAN decoding algorithm."""

    def __init__(self, iterations=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._iterations = iterations
        self.decoder = RCSCANDecoder(
            mask=self.polar_mask,
            is_systematic=self.is_systematic
        )

    def decode(self, received_message):
        """Decode Polar code with Fast SSC decoding algorithm."""
        return self._rc_scan_decode(received_message)

    @property
    def I(self):
        return self._iterations

    def _rc_scan_decode(self, llr_estimated_message):
        """RC SCAN decoding."""
        self.decoder.clean_before_decoding()

        for i in range(self._iterations):
            self.decoder.set_initial_state(llr_estimated_message)
            self.decoder()

            if not self.is_crc_aided:
                continue

            # Validate the result using CRC
            result = self._extract(self.decoder.result)
            if self._check_crc(result):
                return result

        return self._extract(self.decoder.result)

    def _get_extra_params(self):
        return {
            'iterations': self.I,
        }
