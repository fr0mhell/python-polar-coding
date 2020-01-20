from python_polar_coding.polar_codes.fast_ssc import FastSSCPolarCode

from .base import BasicVerifyPolarCodeTestCase


class TestFastSSCCode_1024_512(BasicVerifyPolarCodeTestCase):
    messages = 1000
    polar_code_class = FastSSCPolarCode
    code_parameters = {
        'N': 1024,
        'K': 512,
    }


class TestFastSSCCode_1024_256(BasicVerifyPolarCodeTestCase):
    code_parameters = {
        'N': 1024,
        'K': 256,
    }


class TestFastSSCCode_1024_768(BasicVerifyPolarCodeTestCase):
    code_parameters = {
        'N': 1024,
        'K': 768,
    }


class TestFastSSCCode_2048_512(BasicVerifyPolarCodeTestCase):
    code_parameters = {
        'N': 2048,
        'K': 512,
    }


class TestFastSSCCode_2048_1024(BasicVerifyPolarCodeTestCase):
    code_parameters = {
        'N': 2048,
        'K': 1024,
    }


class TestFastSSCCode_2048_1536(BasicVerifyPolarCodeTestCase):
    code_parameters = {
        'N': 2048,
        'K': 1536,
    }
