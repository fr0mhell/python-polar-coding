from unittest import TestCase

import numpy as np

from polar_codes import SCPolarCode

from .mixins import BasePolarCodeTestMixin
from .utils import SimpleBPSKModAWGNChannel


class TestSCPolarCode(BasePolarCodeTestMixin, TestCase):
    messages = 10000
    codec_class = SCPolarCode
    channel_class = SimpleBPSKModAWGNChannel

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.N = 256
        cls.K = 128
        cls.codec = cls.codec_class(
            codeword_length=cls.N,
            info_length=cls.K,
            is_systematic=True,
        )
        cls.snr_range = np.array([i / 2 for i in range(11)])

    def test_sc_decoder_without_noise(self):
        self._base_test()

    def test_snr_0_0_db(self):
        snd_db = 0.0
        self._base_test(snd_db, with_noise=True)

    def test_snr_0_5_db(self):
        snd_db = 0.5
        self._base_test(snd_db, with_noise=True)

    def test_snr_1_0_db(self):
        snd_db = 1.0
        self._base_test(snd_db, with_noise=True)

    def test_snr_1_5_db(self):
        snd_db = 1.5
        self._base_test(snd_db, with_noise=True)

    def test_snr_2_0_db(self):
        snd_db = 2.0
        self._base_test(snd_db, with_noise=True)

    def test_snr_2_5_db(self):
        snd_db = 2.5
        self._base_test(snd_db, with_noise=True)

    def test_snr_3_0_db(self):
        snd_db = 3.0
        self._base_test(snd_db, with_noise=True)

    def test_snr_3_5_db(self):
        snd_db = 3.5
        self._base_test(snd_db, with_noise=True)

    def test_snr_4_0_db(self):
        snd_db = 4.0
        self._base_test(snd_db, with_noise=True)

    def test_snr_4_5_db(self):
        snd_db = 4.5
        self._base_test(snd_db, with_noise=True)

    def test_snr_5_0_db(self):
        snd_db = 5.0
        self._base_test(snd_db, with_noise=True)
