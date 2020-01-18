from python_polar_coding.polar_codes import FastSSCPolarCode

from .sc import VerifySystematicSCCode


class VerifyFastSSCCode(VerifySystematicSCCode):
    messages = 10000
    firestore_dump = False
    codec_class = FastSSCPolarCode


class TestSystematicSCCode_1024_512(VerifyFastSSCCode):
    code_parameters = {
        'N': 1024,
        'K': 512,
        'is_systematic': True
    }


class TestSystematicSCCode_1024_256(VerifyFastSSCCode):
    code_parameters = {
        'N': 1024,
        'K': 256,
        'is_systematic': True
    }


class TestSystematicSCCode_1024_768(VerifyFastSSCCode):
    code_parameters = {
        'N': 1024,
        'K': 768,
        'is_systematic': True
    }


class TestSystematicSCCode_2048_512(VerifyFastSSCCode):
    code_parameters = {
        'N': 2048,
        'K': 512,
        'is_systematic': True
    }


class TestSystematicSCCode_2048_1024(VerifyFastSSCCode):
    code_parameters = {
        'N': 2048,
        'K': 1024,
        'is_systematic': True
    }


class TestSystematicSCCode_2048_1536(VerifyFastSSCCode):
    code_parameters = {
        'N': 2048,
        'K': 1536,
        'is_systematic': True
    }


class TestSystematicSCCode_8192_4096(VerifyFastSSCCode):
    code_parameters = {
        'N': 8192,
        'K': 4096,
        'design_snr': 1.4,
        'is_systematic': True,
    }
