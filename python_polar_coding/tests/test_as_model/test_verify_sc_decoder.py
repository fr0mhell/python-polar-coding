from .sc import VerifySystematicSCCode


class TestSystematicSCCode_1024_512(VerifySystematicSCCode):
    code_parameters = {
        'N': 1024,
        'K': 512,
        'is_systematic': True
    }


class TestSystematicSCCode_1024_256(VerifySystematicSCCode):
    code_parameters = {
        'N': 1024,
        'K': 256,
        'is_systematic': True
    }


class TestSystematicSCCode_1024_768(VerifySystematicSCCode):
    code_parameters = {
        'N': 1024,
        'K': 768,
        'is_systematic': True
    }


class TestSystematicSCCode_2048_512(VerifySystematicSCCode):
    code_parameters = {
        'N': 2048,
        'K': 512,
        'is_systematic': True
    }


class TestSystematicSCCode_2048_1024(VerifySystematicSCCode):
    code_parameters = {
        'N': 2048,
        'K': 1024,
        'is_systematic': True
    }


class TestSystematicSCCode_2048_1536(VerifySystematicSCCode):
    code_parameters = {
        'N': 2048,
        'K': 1536,
        'is_systematic': True
    }
