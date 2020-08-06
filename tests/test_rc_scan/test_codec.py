from unittest import TestCase

from python_polar_coding.polar_codes.rc_scan import RCSCANPolarCodec
from tests.base import BasicVerifyPolarCode


class TestRCSCANCode_1024_512(BasicVerifyPolarCode, TestCase):
    polar_code_class = RCSCANPolarCodec
    code_parameters = {
        'N': 1024,
        'K': 512,
        'I': 1,
    }


class TestRCSCANCode_1024_256(BasicVerifyPolarCode, TestCase):
    polar_code_class = RCSCANPolarCodec
    code_parameters = {
        'N': 1024,
        'K': 256,
        'I': 1,
    }


class TestRCSCANCode_1024_768(BasicVerifyPolarCode, TestCase):
    polar_code_class = RCSCANPolarCodec
    code_parameters = {
        'N': 1024,
        'K': 768,
        'I': 1,
    }


# Iterations 2


class TestRCSCANCode_1024_512_iter_2(BasicVerifyPolarCode, TestCase):
    polar_code_class = RCSCANPolarCodec
    code_parameters = {
        'N': 1024,
        'K': 512,
        'I': 2,
    }


class TestRCSCANCode_1024_256_iter_2(BasicVerifyPolarCode, TestCase):
    polar_code_class = RCSCANPolarCodec
    code_parameters = {
        'N': 1024,
        'K': 256,
        'I': 2,
    }


class TestRCSCANCode_1024_768_iter_2(BasicVerifyPolarCode, TestCase):
    polar_code_class = RCSCANPolarCodec
    code_parameters = {
        'N': 1024,
        'K': 768,
        'I': 2,
    }


# Iterations 4


class TestRCSCANCode_1024_512_iter_4(BasicVerifyPolarCode, TestCase):
    polar_code_class = RCSCANPolarCodec
    code_parameters = {
        'N': 1024,
        'K': 512,
        'I': 4,
    }


class TestRCSCANCode_1024_256_iter_4(BasicVerifyPolarCode, TestCase):
    polar_code_class = RCSCANPolarCodec
    code_parameters = {
        'N': 1024,
        'K': 256,
        'I': 4,
    }


class TestRCSCANCode_1024_768_iter_4(BasicVerifyPolarCode, TestCase):
    polar_code_class = RCSCANPolarCodec
    code_parameters = {
        'N': 1024,
        'K': 768,
        'I': 4,
    }
