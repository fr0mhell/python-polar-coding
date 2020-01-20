from python_polar_coding.polar_codes.sc_list import SCListPolarCodeWithCRC

from .base import BasicVerifyPolarCodeTestCase


class TestSCListPolarCode1024_512_4(BasicVerifyPolarCodeTestCase):
    messages = 1000
    polar_code_class = SCListPolarCodeWithCRC
    code_parameters = {
        'N': 1024,
        'K': 512,
        'L': 4,
        'crc_size': 32,
    }


class TestSCListPolarCode1024_512_8(BasicVerifyPolarCodeTestCase):
    messages = 1000
    polar_code_class = SCListPolarCodeWithCRC
    code_parameters = {
        'N': 1024,
        'K': 512,
        'L': 8,
    }


class TestSCListPolarCode2048_1024_8(BasicVerifyPolarCodeTestCase):
    messages = 1000
    polar_code_class = SCListPolarCodeWithCRC
    code_parameters = {
        'N': 2048,
        'K': 1024,
        'L': 8,
        'crc_size': 32,
    }


class TestSCListPolarCode2048_1024_16(BasicVerifyPolarCodeTestCase):
    messages = 1000
    polar_code_class = SCListPolarCodeWithCRC
    code_parameters = {
        'N': 2048,
        'K': 1024,
        'L': 16,
        'crc_size': 32,
    }


class TestSCListPolarCode2048_1024_32(BasicVerifyPolarCodeTestCase):
    messages = 1000
    polar_code_class = SCListPolarCodeWithCRC
    code_parameters = {
        'N': 2048,
        'K': 1024,
        'L': 32,
        'crc_size': 32,
    }
