from polar_codes import SCListPolarCode

from .base import VerificationChannelTestCase


class VerifySystematicSCListCode(VerificationChannelTestCase):
    messages = 10000
    codec_class = SCListPolarCode

    @classmethod
    def _get_filename(cls):
        N = cls.code_parameters['codeword_length']
        K = cls.code_parameters['info_length']
        L = cls.code_parameters['list_size']
        filename = f'{N}_{K}_L_{L}'
        if cls.code_parameters.get('is_crc_aided'):
            filename += '_crc'
        return f'{filename}.json'

    def test_sc_decoder_without_noise(self):
        pass

    def test_snr_1_0_db(self):
        snd_db = 1.0
        self._base_test(snd_db)

    def test_snr_1_25_db(self):
        snd_db = 1.25
        self._base_test(snd_db)

    def test_snr_1_5_db(self):
        snd_db = 1.5
        self._base_test(snd_db)

    def test_snr_1_75_db(self):
        snd_db = 1.75
        self._base_test(snd_db)

    def test_snr_2_0_db(self):
        snd_db = 2.0
        self._base_test(snd_db)

    def test_snr_2_25_db(self):
        snd_db = 2.25
        self._base_test(snd_db)

    def test_snr_2_5_db(self):
        snd_db = 2.5
        self._base_test(snd_db)

    def test_snr_2_75_db(self):
        snd_db = 2.75
        self._base_test(snd_db)

    def test_snr_3_0_db(self):
        snd_db = 3.0
        self._base_test(snd_db)
