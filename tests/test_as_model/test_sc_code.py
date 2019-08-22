from unittest import TestCase

from polar_codes import SCPolarCode

from .base import BasePolarCodeTestMixin
from .channels import SimpleBPSKModAWGNChannel


class TestSCPolarCode(BasePolarCodeTestMixin, TestCase):
    """Example of how to model Polar Codes using unittest."""
    messages = 10000
    codec_class = SCPolarCode
    channel_class = SimpleBPSKModAWGNChannel
    code_parameters = {
        'codeword_length': 256,
        'info_length': 128,
        'is_systematic': True
    }

    def test_snr_0_0_db(self):
        snd_db = 0.0
        self._base_test(snd_db)

    def test_snr_0_5_db(self):
        snd_db = 0.5
        self._base_test(snd_db)

    def test_snr_1_0_db(self):
        snd_db = 1.0
        self._base_test(snd_db)

    def test_snr_1_5_db(self):
        snd_db = 1.5
        self._base_test(snd_db)

    def test_snr_2_0_db(self):
        snd_db = 2.0
        self._base_test(snd_db)

    def test_snr_2_5_db(self):
        snd_db = 2.5
        self._base_test(snd_db)

    def test_snr_3_0_db(self):
        snd_db = 3.0
        self._base_test(snd_db)

    def test_snr_3_5_db(self):
        snd_db = 3.5
        self._base_test(snd_db)

    def test_snr_4_0_db(self):
        snd_db = 4.0
        self._base_test(snd_db)
