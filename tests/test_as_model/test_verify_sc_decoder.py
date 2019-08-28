from .sc import VerifySystematicSCCode


class TestSystematicSCCode_1024_512(VerifySystematicSCCode):
    code_parameters = {
        'codeword_length': 1024,
        'info_length': 512,
        'is_systematic': True
    }


class TestSystematicSCCode_1024_256(VerifySystematicSCCode):
    code_parameters = {
        'codeword_length': 1024,
        'info_length': 256,
        'is_systematic': True
    }


class TestSystematicSCCode_1024_768(VerifySystematicSCCode):
    code_parameters = {
        'codeword_length': 1024,
        'info_length': 768,
        'is_systematic': True
    }


class TestSystematicSCCode_2048_512(VerifySystematicSCCode):
    code_parameters = {
        'codeword_length': 2048,
        'info_length': 512,
        'is_systematic': True
    }


class TestSystematicSCCode_2048_1024(VerifySystematicSCCode):
    code_parameters = {
        'codeword_length': 2048,
        'info_length': 1024,
        'is_systematic': True
    }


class TestSystematicSCCode_2048_1536(VerifySystematicSCCode):
    code_parameters = {
        'codeword_length': 2048,
        'info_length': 1536,
        'is_systematic': True
    }
