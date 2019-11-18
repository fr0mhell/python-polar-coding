from polar_codes import SCPolarCode

from .base import VerificationChannelTestCase


class VerifySystematicSCCode(VerificationChannelTestCase):
    messages = 10000
    codec_class = SCPolarCode

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
