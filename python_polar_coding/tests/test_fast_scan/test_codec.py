from unittest import TestCase

from python_polar_coding.polar_codes.fast_scan import FastSCANCodec
from python_polar_coding.tests.base import BasicVerifyPolarCode


class TestFastSCANCodec_1024_512(BasicVerifyPolarCode, TestCase):
    polar_code_class = FastSCANCodec
    code_parameters = {
        'N': 1024,
        'K': 512,
        'I': 1,
    }


class TestFastSCANCodec_1024_256(BasicVerifyPolarCode, TestCase):
    polar_code_class = FastSCANCodec
    code_parameters = {
        'N': 1024,
        'K': 256,
        'I': 1,
    }


class TestFastSCANCodec_1024_768(BasicVerifyPolarCode, TestCase):
    polar_code_class = FastSCANCodec
    code_parameters = {
        'N': 1024,
        'K': 768,
        'I': 1,
    }


class TestFastSCANCodec_2048_512(BasicVerifyPolarCode, TestCase):
    polar_code_class = FastSCANCodec
    code_parameters = {
        'N': 2048,
        'K': 512,
        'I': 1,
    }


class TestFastSCANCodec_2048_1024(BasicVerifyPolarCode, TestCase):
    polar_code_class = FastSCANCodec
    code_parameters = {
        'N': 2048,
        'K': 1024,
        'I': 1,
    }


class TestFastSCANCodec_2048_1536(BasicVerifyPolarCode, TestCase):
    polar_code_class = FastSCANCodec
    code_parameters = {
        'N': 2048,
        'K': 1536,
        'I': 1,
    }


# Iterations 2


class TestFastSCANCodec_1024_512_iter_2(BasicVerifyPolarCode, TestCase):
    polar_code_class = FastSCANCodec
    code_parameters = {
        'N': 1024,
        'K': 512,
        'I': 2,
    }


class TestFastSCANCodec_1024_256_iter_2(BasicVerifyPolarCode, TestCase):
    polar_code_class = FastSCANCodec
    code_parameters = {
        'N': 1024,
        'K': 256,
        'I': 2,
    }


class TestFastSCANCodec_1024_768_iter_2(BasicVerifyPolarCode, TestCase):
    polar_code_class = FastSCANCodec
    code_parameters = {
        'N': 1024,
        'K': 768,
        'I': 2,
    }


class TestFastSCANCodec_2048_512_iter_2(BasicVerifyPolarCode, TestCase):
    polar_code_class = FastSCANCodec
    code_parameters = {
        'N': 2048,
        'K': 512,
        'I': 2,
    }


class TestFastSCANCodec_2048_1024_iter_2(BasicVerifyPolarCode, TestCase):
    polar_code_class = FastSCANCodec
    code_parameters = {
        'N': 2048,
        'K': 1024,
        'I': 2,
    }


class TestFastSCANCodec_2048_1536_iter_2(BasicVerifyPolarCode, TestCase):
    polar_code_class = FastSCANCodec
    code_parameters = {
        'N': 2048,
        'K': 1536,
        'I': 2,
    }


# Iterations 4


class TestFastSCANCodec_1024_512_iter_4(BasicVerifyPolarCode, TestCase):
    polar_code_class = FastSCANCodec
    code_parameters = {
        'N': 1024,
        'K': 512,
        'I': 4,
    }


class TestFastSCANCodec_1024_256_iter_4(BasicVerifyPolarCode, TestCase):
    polar_code_class = FastSCANCodec
    code_parameters = {
        'N': 1024,
        'K': 256,
        'I': 4,
    }


class TestFastSCANCodec_1024_768_iter_4(BasicVerifyPolarCode, TestCase):
    polar_code_class = FastSCANCodec
    code_parameters = {
        'N': 1024,
        'K': 768,
        'I': 4,
    }


class TestFastSCANCodec_2048_512_iter_4(BasicVerifyPolarCode, TestCase):
    polar_code_class = FastSCANCodec
    code_parameters = {
        'N': 2048,
        'K': 512,
        'I': 4,
    }


class TestFastSCANCodec_2048_1024_iter_4(BasicVerifyPolarCode, TestCase):
    polar_code_class = FastSCANCodec
    code_parameters = {
        'N': 2048,
        'K': 1024,
        'I': 4,
    }


class TestFastSCANCodec_2048_1536_iter_4(BasicVerifyPolarCode, TestCase):
    polar_code_class = FastSCANCodec
    code_parameters = {
        'N': 2048,
        'K': 1536,
        'I': 4,
    }
