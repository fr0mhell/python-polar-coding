from unittest import TestCase

from python_polar_coding.polar_codes.g_fast_scan import GFastSCANCodec
from tests.base import BasicVerifyPolarCode

# Iterations 2


class TestGFastSCANCodec_1024_512_iter_2_AF_0(BasicVerifyPolarCode, TestCase):
    polar_code_class = GFastSCANCodec
    code_parameters = {
        'N': 1024,
        'K': 512,
        'I': 2,
        'AF': 0,
    }


class TestGFastSCANCodec_1024_512_iter_2_AF_1(BasicVerifyPolarCode, TestCase):
    polar_code_class = GFastSCANCodec
    code_parameters = {
        'N': 1024,
        'K': 512,
        'I': 2,
        'AF': 1,
    }


class TestGFastSCANCodec_1024_512_iter_2_AF_2(BasicVerifyPolarCode, TestCase):
    polar_code_class = GFastSCANCodec
    code_parameters = {
        'N': 1024,
        'K': 512,
        'I': 2,
        'AF': 2,
    }


class TestGFastSCANCodec_1024_512_iter_2_AF_3(BasicVerifyPolarCode, TestCase):
    polar_code_class = GFastSCANCodec
    code_parameters = {
        'N': 1024,
        'K': 512,
        'I': 2,
        'AF': 3,
    }


class TestGFastSCANCodec_1024_256_iter_2_AF_0(BasicVerifyPolarCode, TestCase):
    polar_code_class = GFastSCANCodec
    code_parameters = {
        'N': 1024,
        'K': 256,
        'I': 2,
        'AF': 0,
    }


class TestGFastSCANCodec_1024_256_iter_2_AF_1(BasicVerifyPolarCode, TestCase):
    polar_code_class = GFastSCANCodec
    code_parameters = {
        'N': 1024,
        'K': 256,
        'I': 2,
        'AF': 1,
    }


class TestGFastSCANCodec_1024_256_iter_2_AF_2(BasicVerifyPolarCode, TestCase):
    polar_code_class = GFastSCANCodec
    code_parameters = {
        'N': 1024,
        'K': 256,
        'I': 2,
        'AF': 2,
    }


class TestGFastSCANCodec_1024_256_iter_2_AF_3(BasicVerifyPolarCode, TestCase):
    polar_code_class = GFastSCANCodec
    code_parameters = {
        'N': 1024,
        'K': 256,
        'I': 2,
        'AF': 3,
    }


class TestGFastSCANCodec_1024_768_iter_2_AF_0(BasicVerifyPolarCode, TestCase):
    polar_code_class = GFastSCANCodec
    code_parameters = {
        'N': 1024,
        'K': 768,
        'I': 2,
        'AF': 0,
    }


class TestGFastSCANCodec_1024_768_iter_2_AF_1(BasicVerifyPolarCode, TestCase):
    polar_code_class = GFastSCANCodec
    code_parameters = {
        'N': 1024,
        'K': 768,
        'I': 2,
        'AF': 1,
    }


class TestGFastSCANCodec_1024_768_iter_2_AF_2(BasicVerifyPolarCode, TestCase):
    polar_code_class = GFastSCANCodec
    code_parameters = {
        'N': 1024,
        'K': 768,
        'I': 2,
        'AF': 2,
    }


class TestGFastSCANCodec_1024_768_iter_2_AF_3(BasicVerifyPolarCode, TestCase):
    polar_code_class = GFastSCANCodec
    code_parameters = {
        'N': 1024,
        'K': 768,
        'I': 2,
        'AF': 3,
    }


# Iterations 4


class TestGFastSCANCodec_1024_512_iter_4_AF_0(BasicVerifyPolarCode, TestCase):
    polar_code_class = GFastSCANCodec
    code_parameters = {
        'N': 1024,
        'K': 512,
        'I': 4,
        'AF': 0,
    }


class TestGFastSCANCodec_1024_512_iter_4_AF_1(BasicVerifyPolarCode, TestCase):
    polar_code_class = GFastSCANCodec
    code_parameters = {
        'N': 1024,
        'K': 512,
        'I': 4,
        'AF': 1,
    }


class TestGFastSCANCodec_1024_512_iter_4_AF_2(BasicVerifyPolarCode, TestCase):
    polar_code_class = GFastSCANCodec
    code_parameters = {
        'N': 1024,
        'K': 512,
        'I': 4,
        'AF': 2,
    }


class TestGFastSCANCodec_1024_512_iter_4_AF_3(BasicVerifyPolarCode, TestCase):
    polar_code_class = GFastSCANCodec
    code_parameters = {
        'N': 1024,
        'K': 512,
        'I': 4,
        'AF': 3,
    }


class TestGFastSCANCodec_1024_256_iter_4_AF_0(BasicVerifyPolarCode, TestCase):
    polar_code_class = GFastSCANCodec
    code_parameters = {
        'N': 1024,
        'K': 256,
        'I': 4,
        'AF': 0,
    }


class TestGFastSCANCodec_1024_256_iter_4_AF_1(BasicVerifyPolarCode, TestCase):
    polar_code_class = GFastSCANCodec
    code_parameters = {
        'N': 1024,
        'K': 256,
        'I': 4,
        'AF': 1,
    }


class TestGFastSCANCodec_1024_256_iter_4_AF_2(BasicVerifyPolarCode, TestCase):
    polar_code_class = GFastSCANCodec
    code_parameters = {
        'N': 1024,
        'K': 256,
        'I': 4,
        'AF': 2,
    }


class TestGFastSCANCodec_1024_256_iter_4_AF_3(BasicVerifyPolarCode, TestCase):
    polar_code_class = GFastSCANCodec
    code_parameters = {
        'N': 1024,
        'K': 256,
        'I': 4,
        'AF': 3,
    }


class TestGFastSCANCodec_1024_768_iter_4_AF_0(BasicVerifyPolarCode, TestCase):
    polar_code_class = GFastSCANCodec
    code_parameters = {
        'N': 1024,
        'K': 768,
        'I': 4,
        'AF': 0,
    }


class TestGFastSCANCodec_1024_768_iter_4_AF_1(BasicVerifyPolarCode, TestCase):
    polar_code_class = GFastSCANCodec
    code_parameters = {
        'N': 1024,
        'K': 768,
        'I': 4,
        'AF': 1,
    }


class TestGFastSCANCodec_1024_768_iter_4_AF_2(BasicVerifyPolarCode, TestCase):
    polar_code_class = GFastSCANCodec
    code_parameters = {
        'N': 1024,
        'K': 768,
        'I': 4,
        'AF': 2,
    }


class TestGFastSCANCodec_1024_768_iter_4_AF_3(BasicVerifyPolarCode, TestCase):
    polar_code_class = GFastSCANCodec
    code_parameters = {
        'N': 1024,
        'K': 768,
        'I': 4,
        'AF': 3,
    }