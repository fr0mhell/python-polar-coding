from .sc_list import VerifySystematicSCListCode


class TestSCListPolarCode1024_512_2(VerifySystematicSCListCode):
    code_parameters = {
        'N': 1024,
        'K': 512,
        'is_systematic': True,
        'L': 2,
    }


class TestSCListPolarCode1024_512_4(VerifySystematicSCListCode):
    code_parameters = {
        'N': 1024,
        'K': 512,
        'is_systematic': True,
        'L': 4,
    }


class TestSCListPolarCode1024_512_4_crc(VerifySystematicSCListCode):
    code_parameters = {
        'N': 1024,
        'K': 512,
        'is_systematic': True,
        'L': 4,
        'crc_size': 32,
    }


class TestSCListPolarCode1024_512_8(VerifySystematicSCListCode):
    code_parameters = {
        'N': 1024,
        'K': 512,
        'is_systematic': True,
        'L': 8,
    }


class TestSCListPolarCode1024_512_8_crc(VerifySystematicSCListCode):
    code_parameters = {
        'N': 1024,
        'K': 512,
        'is_systematic': True,
        'L': 8,
        'crc_size': 32,
    }


class TestSCListPolarCode2048_1024_8(VerifySystematicSCListCode):
    code_parameters = {
        'N': 2048,
        'K': 1024,
        'is_systematic': True,
        'L': 8,
    }


class TestSCListPolarCode2048_1024_8_crc(VerifySystematicSCListCode):
    code_parameters = {
        'N': 2048,
        'K': 1024,
        'is_systematic': True,
        'L': 8,
        'crc_size': 32,
    }


class TestSCListPolarCode2048_1024_32(VerifySystematicSCListCode):
    code_parameters = {
        'N': 2048,
        'K': 1024,
        'is_systematic': True,
        'L': 32,
    }


class TestSCListPolarCode2048_1024_32_crc(VerifySystematicSCListCode):
    code_parameters = {
        'N': 2048,
        'K': 1024,
        'is_systematic': True,
        'L': 32,
        'crc_size': 32,
    }
