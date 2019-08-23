from unittest import TestCase

from polar_codes import SCListPolarCode, SCPolarCode

from .base import BasePolarCodeTestMixin
from .channels import SimpleBPSKModAWGNChannel


class TestSCPolarCode2048_1024(BasePolarCodeTestMixin, TestCase):
    messages = 10000
    codec_class = SCPolarCode
    channel_class = SimpleBPSKModAWGNChannel
    code_parameters = {
        'codeword_length': 2048,
        'info_length': 1024,
        'design_snr': 2.0,
        'is_systematic': True,
    }

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


class TestSCListPolarCode2048_1024_8(TestSCPolarCode2048_1024):
    codec_class = SCListPolarCode
    code_parameters = {
        'codeword_length': 2048,
        'info_length': 1024,
        'design_snr': 2.0,
        'is_systematic': True,
        'list_size': 8,
    }

    @classmethod
    def _get_filename(cls):
        N = cls.code_parameters['codeword_length']
        K = cls.code_parameters['info_length']
        L = cls.code_parameters['list_size']
        filename = f'{N}_{K}_L_{L}'
        if cls.code_parameters.get('is_crc_aided'):
            filename += '_crc'
        return f'{filename}.json'


class TestSCListPolarCode2048_1024_8_crc(TestSCListPolarCode2048_1024_8):
    code_parameters = {
        'codeword_length': 2048,
        'info_length': 1024,
        'design_snr': 2.0,
        'is_systematic': True,
        'list_size': 8,
        'is_crc_aided': True,
    }


class TestSCListPolarCode2048_1024_32(TestSCListPolarCode2048_1024_8):
    code_parameters = {
        'codeword_length': 2048,
        'info_length': 1024,
        'design_snr': 2.0,
        'is_systematic': True,
        'list_size': 32,
    }


class TestSCListPolarCode2048_1024_32_crc(TestSCListPolarCode2048_1024_8):
    code_parameters = {
        'codeword_length': 2048,
        'info_length': 1024,
        'design_snr': 2.0,
        'is_systematic': True,
        'list_size': 32,
        'is_crc_aided': True,
    }
