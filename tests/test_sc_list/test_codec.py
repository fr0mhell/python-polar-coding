from unittest import TestCase

from python_polar_coding.polar_codes.sc_list import SCListPolarCodec
from tests.base import BasicVerifyPolarCode


class TestSCListPolarCode1024_512_4(BasicVerifyPolarCode, TestCase):
    polar_code_class = SCListPolarCodec
    code_parameters = {
        'N': 1024,
        'K': 512,
        'L': 4,
    }


class TestSCListPolarCode1024_512_8(BasicVerifyPolarCode, TestCase):
    polar_code_class = SCListPolarCodec
    code_parameters = {
        'N': 1024,
        'K': 512,
        'L': 8,
    }
