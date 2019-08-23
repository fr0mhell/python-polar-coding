from .base import BasicPolarCode
from .decoders import SCListDecoder


class SCListPolarCode(BasicPolarCode):
    """Polar code with SC decoding algorithm."""
    def __init__(self, list_size=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.list_size = list_size
        self.decoder = SCListDecoder(
            mask=self.polar_mask,
            is_systematic=self.is_systematic,
            list_size=self.list_size,
        )

    def _get_extra_params(self):
        return {
            'list_size': self.L,
        }

    @property
    def L(self):
        return self.list_size

    def decode(self, received_message):
        """Decode Polar code with SC List decoding algorithm."""
        return self._sc_list_decode(received_message)

    def _sc_list_decode(self, llr_estimated_message):
        """Successive cancellation list decoder

        Based on: https://arxiv.org/abs/1206.0050.

        """
        self.decoder.initialize(llr_estimated_message)

        for step in range(self.N):
            self.decoder(step)

        # If the code is not CRC-aided, return the result of the best path
        if not self.is_crc_aided:
            best_result = self.decoder.paths[0].result
            return self._extract(best_result)

        return self._get_correct_result_using_crc()

    def _get_correct_result_using_crc(self):
        """Get correct result from SC paths using CRC.

        If no correct paths found, get the result from best path.

        """
        for path in self.decoder.paths:
            result = self._extract(path.result)
            if self._check_crc(result):
                return self._remove_crc(result)

        return self._extract(self.decoder.paths[0].result)
