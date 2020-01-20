from python_polar_coding.polar_codes.sc_list import SCListPolarCode

from .base import BasicVerifyPolarCodeTestCase


class TestSCListPolarCode1024_512_4(BasicVerifyPolarCodeTestCase):
    messages = 1000
    polar_code_class = SCListPolarCode
    code_parameters = {
        'N': 1024,
        'K': 512,
        'L': 4,
    }


class TestSCListPolarCode1024_512_8(BasicVerifyPolarCodeTestCase):
    messages = 1000
    polar_code_class = SCListPolarCode
    code_parameters = {
        'N': 1024,
        'K': 512,
        'L': 8,
    }


class TestSCListPolarCode2048_1024_8(BasicVerifyPolarCodeTestCase):
    messages = 1000
    polar_code_class = SCListPolarCode
    code_parameters = {
        'N': 2048,
        'K': 1024,
        'L': 8,
    }


class TestSCListPolarCode2048_1024_16(BasicVerifyPolarCodeTestCase):
    messages = 1000
    polar_code_class = SCListPolarCode
    code_parameters = {
        'N': 2048,
        'K': 1024,
        'L': 16,
    }


class TestSCListPolarCode2048_1024_32(BasicVerifyPolarCodeTestCase):
    messages = 1000
    polar_code_class = SCListPolarCode
    code_parameters = {
        'N': 2048,
        'K': 1024,
        'L': 32,
    }
