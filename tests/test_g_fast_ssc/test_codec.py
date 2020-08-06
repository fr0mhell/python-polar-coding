from unittest import TestCase

from python_polar_coding.polar_codes.g_fast_ssc import GFastSSCPolarCodec
from tests.base import BasicVerifyPolarCode


class TestGeneralizedFastSSCCode_1024_256_AF_0(BasicVerifyPolarCode, TestCase):
    polar_code_class = GFastSSCPolarCodec
    code_parameters = {
        'N': 1024,
        'K': 256,
        'AF': 0,
    }


class TestGeneralizedFastSSCCode_1024_256_AF_1(BasicVerifyPolarCode, TestCase):
    polar_code_class = GFastSSCPolarCodec
    code_parameters = {
        'N': 1024,
        'K': 256,
        'AF': 1,
    }


class TestGeneralizedFastSSCCode_1024_256_AF_2(BasicVerifyPolarCode, TestCase):
    polar_code_class = GFastSSCPolarCodec
    code_parameters = {
        'N': 1024,
        'K': 256,
        'AF': 2,
    }


class TestGeneralizedFastSSCCode_1024_256_AF_3(BasicVerifyPolarCode, TestCase):
    polar_code_class = GFastSSCPolarCodec
    code_parameters = {
        'N': 1024,
        'K': 256,
        'AF': 3,
    }


class TestGeneralizedFastSSCCode_1024_512_AF_0(BasicVerifyPolarCode, TestCase):
    polar_code_class = GFastSSCPolarCodec
    code_parameters = {
        'N': 1024,
        'K': 512,
        'AF': 0,
    }


class TestGeneralizedFastSSCCode_1024_512_AF_1(BasicVerifyPolarCode, TestCase):
    polar_code_class = GFastSSCPolarCodec
    code_parameters = {
        'N': 1024,
        'K': 512,
        'AF': 1,
    }


class TestGeneralizedFastSSCCode_1024_512_AF_2(BasicVerifyPolarCode, TestCase):
    polar_code_class = GFastSSCPolarCodec
    code_parameters = {
        'N': 1024,
        'K': 512,
        'AF': 2,
    }


class TestGeneralizedFastSSCCode_1024_512_AF_3(BasicVerifyPolarCode, TestCase):
    polar_code_class = GFastSSCPolarCodec
    code_parameters = {
        'N': 1024,
        'K': 512,
        'AF': 3,
    }


class TestGeneralizedFastSSCCode_1024_768_AF_0(BasicVerifyPolarCode, TestCase):
    polar_code_class = GFastSSCPolarCodec
    code_parameters = {
        'N': 1024,
        'K': 768,
        'AF': 0,
    }


class TestGeneralizedFastSSCCode_1024_768_AF_1(BasicVerifyPolarCode, TestCase):
    polar_code_class = GFastSSCPolarCodec
    code_parameters = {
        'N': 1024,
        'K': 768,
        'AF': 1,
    }


class TestGeneralizedFastSSCCode_1024_768_AF_2(BasicVerifyPolarCode, TestCase):
    polar_code_class = GFastSSCPolarCodec
    code_parameters = {
        'N': 1024,
        'K': 768,
        'AF': 2,
    }


class TestGeneralizedFastSSCCode_1024_768_AF_3(BasicVerifyPolarCode, TestCase):
    polar_code_class = GFastSSCPolarCodec
    code_parameters = {
        'N': 1024,
        'K': 768,
        'AF': 3,
    }
