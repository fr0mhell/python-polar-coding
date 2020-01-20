from python_polar_coding.polar_codes.rc_scan import RCSCANPolarCode

from .base import BasicVerifyPolarCodeTestCase


class TestRCSCANCode_1024_512(BasicVerifyPolarCodeTestCase):
    messages = 1000
    polar_code_class = RCSCANPolarCode
    code_parameters = {
        'N': 1024,
        'K': 512,
        'I': 1,
    }


class TestRCSCANCode_1024_256(BasicVerifyPolarCodeTestCase):
    messages = 1000
    polar_code_class = RCSCANPolarCode
    code_parameters = {
        'N': 1024,
        'K': 256,
        'I': 1,
    }


class TestRCSCANCode_1024_768(BasicVerifyPolarCodeTestCase):
    messages = 1000
    polar_code_class = RCSCANPolarCode
    code_parameters = {
        'N': 1024,
        'K': 768,
        'I': 1,
    }


class TestRCSCANCode_2048_512(BasicVerifyPolarCodeTestCase):
    messages = 1000
    polar_code_class = RCSCANPolarCode
    code_parameters = {
        'N': 2048,
        'K': 512,
        'I': 1,
    }


class TestRCSCANCode_2048_1024(BasicVerifyPolarCodeTestCase):
    messages = 1000
    polar_code_class = RCSCANPolarCode
    code_parameters = {
        'N': 2048,
        'K': 1024,
        'I': 1,
    }


class TestRCSCANCode_2048_1536(BasicVerifyPolarCodeTestCase):
    messages = 1000
    polar_code_class = RCSCANPolarCode
    code_parameters = {
        'N': 2048,
        'K': 1536,
        'I': 1,
    }


# Iterations 2


class TestRCSCANCode_1024_512_iter_2(BasicVerifyPolarCodeTestCase):
    messages = 1000
    polar_code_class = RCSCANPolarCode
    code_parameters = {
        'N': 1024,
        'K': 512,
        'I': 2,
    }


class TestRCSCANCode_1024_256_iter_2(BasicVerifyPolarCodeTestCase):
    messages = 1000
    polar_code_class = RCSCANPolarCode
    code_parameters = {
        'N': 1024,
        'K': 256,
        'I': 2,
    }


class TestRCSCANCode_1024_768_iter_2(BasicVerifyPolarCodeTestCase):
    messages = 1000
    polar_code_class = RCSCANPolarCode
    code_parameters = {
        'N': 1024,
        'K': 768,
        'I': 2,
    }


class TestRCSCANCode_2048_512_iter_2(BasicVerifyPolarCodeTestCase):
    messages = 1000
    polar_code_class = RCSCANPolarCode
    code_parameters = {
        'N': 2048,
        'K': 512,
        'I': 2,
    }


class TestRCSCANCode_2048_1024_iter_2(BasicVerifyPolarCodeTestCase):
    messages = 1000
    polar_code_class = RCSCANPolarCode
    code_parameters = {
        'N': 2048,
        'K': 1024,
        'I': 2,
    }


class TestRCSCANCode_2048_1536_iter_2(BasicVerifyPolarCodeTestCase):
    messages = 1000
    polar_code_class = RCSCANPolarCode
    code_parameters = {
        'N': 2048,
        'K': 1536,
        'I': 2,
    }


# Iterations 4


class TestRCSCANCode_1024_512_iter_4(BasicVerifyPolarCodeTestCase):
    messages = 1000
    polar_code_class = RCSCANPolarCode
    code_parameters = {
        'N': 1024,
        'K': 512,
        'I': 4,
    }


class TestRCSCANCode_1024_256_iter_4(BasicVerifyPolarCodeTestCase):
    messages = 1000
    polar_code_class = RCSCANPolarCode
    code_parameters = {
        'N': 1024,
        'K': 256,
        'I': 4,
    }


class TestRCSCANCode_1024_768_iter_4(BasicVerifyPolarCodeTestCase):
    messages = 1000
    polar_code_class = RCSCANPolarCode
    code_parameters = {
        'N': 1024,
        'K': 768,
        'I': 4,
    }


class TestRCSCANCode_2048_512_iter_4(BasicVerifyPolarCodeTestCase):
    messages = 1000
    polar_code_class = RCSCANPolarCode
    code_parameters = {
        'N': 2048,
        'K': 512,
        'I': 4,
    }


class TestRCSCANCode_2048_1024_iter_4(BasicVerifyPolarCodeTestCase):
    messages = 1000
    polar_code_class = RCSCANPolarCode
    code_parameters = {
        'N': 2048,
        'K': 1024,
        'I': 4,
    }


class TestRCSCANCode_2048_1536_iter_4(BasicVerifyPolarCodeTestCase):
    messages = 1000
    polar_code_class = RCSCANPolarCode
    code_parameters = {
        'N': 2048,
        'K': 1536,
        'I': 4,
    }
