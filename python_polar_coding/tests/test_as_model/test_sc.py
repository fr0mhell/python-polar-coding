from python_polar_coding.polar_codes.sc import SCPolarCode

from .base import BasicVerifyPolarCodeTestCase


class TestSCCode_1024_512(BasicVerifyPolarCodeTestCase):
    messages = 1000
    polar_code_class = SCPolarCode
    code_parameters = {
        'N': 1024,
        'K': 512,
    }


class TestSCCode_1024_256(BasicVerifyPolarCodeTestCase):
    messages = 1000
    polar_code_class = SCPolarCode
    code_parameters = {
        'N': 1024,
        'K': 256,
    }


class TestSCCode_1024_768(BasicVerifyPolarCodeTestCase):
    messages = 1000
    polar_code_class = SCPolarCode
    code_parameters = {
        'N': 1024,
        'K': 768,
    }


class TestSCCode_2048_512(BasicVerifyPolarCodeTestCase):
    messages = 1000
    polar_code_class = SCPolarCode
    code_parameters = {
        'N': 2048,
        'K': 512,
    }


class TestSCCode_2048_1024(BasicVerifyPolarCodeTestCase):
    messages = 1000
    polar_code_class = SCPolarCode
    code_parameters = {
        'N': 2048,
        'K': 1024,
    }


class TestSCCode_2048_1536(BasicVerifyPolarCodeTestCase):
    messages = 1000
    polar_code_class = SCPolarCode
    code_parameters = {
        'N': 2048,
        'K': 1536,
    }
