from polar_codes import FastSSCPolarCode

from .sc import VerifySystematicSCCode


class VerifyFastSSCCode(VerifySystematicSCCode):
    messages = 10000
    firestore_dump = False
    codec_class = FastSSCPolarCode


class TestSystematicSCCode_1024_512(VerifyFastSSCCode):
    code_parameters = {
        'codeword_length': 1024,
        'info_length': 512,
        'is_systematic': True
    }


class TestSystematicSCCode_1024_256(VerifyFastSSCCode):
    code_parameters = {
        'codeword_length': 1024,
        'info_length': 256,
        'is_systematic': True
    }


class TestSystematicSCCode_1024_768(VerifyFastSSCCode):
    code_parameters = {
        'codeword_length': 1024,
        'info_length': 768,
        'is_systematic': True
    }


class TestSystematicSCCode_2048_512(VerifyFastSSCCode):
    code_parameters = {
        'codeword_length': 2048,
        'info_length': 512,
        'is_systematic': True
    }


class TestSystematicSCCode_2048_1024(VerifyFastSSCCode):
    code_parameters = {
        'codeword_length': 2048,
        'info_length': 1024,
        'is_systematic': True
    }


class TestSystematicSCCode_2048_1536(VerifyFastSSCCode):
    code_parameters = {
        'codeword_length': 2048,
        'info_length': 1536,
        'is_systematic': True
    }


class TestSystematicSCCode_8192_4096(VerifyFastSSCCode):
    code_parameters = {
        'codeword_length': 8192,
        'info_length': 4096,
        'design_snr': 1.4,
        'is_systematic': True,
    }
