from .sc_list import VerifySystematicSCListCode


class TestSCListPolarCode1024_512_2(VerifySystematicSCListCode):
    code_parameters = {
        'codeword_length': 1024,
        'info_length': 512,
        'is_systematic': True,
        'list_size': 2,
    }


class TestSCListPolarCode1024_512_4(VerifySystematicSCListCode):
    code_parameters = {
        'codeword_length': 1024,
        'info_length': 512,
        'is_systematic': True,
        'list_size': 4,
    }


class TestSCListPolarCode1024_512_4_crc(VerifySystematicSCListCode):
    code_parameters = {
        'codeword_length': 1024,
        'info_length': 512,
        'is_systematic': True,
        'list_size': 4,
        'is_crc_aided': True,
    }


class TestSCListPolarCode1024_512_8(VerifySystematicSCListCode):
    code_parameters = {
        'codeword_length': 1024,
        'info_length': 512,
        'is_systematic': True,
        'list_size': 8,
    }


class TestSCListPolarCode1024_512_8_crc(VerifySystematicSCListCode):
    code_parameters = {
        'codeword_length': 1024,
        'info_length': 512,
        'is_systematic': True,
        'list_size': 8,
        'is_crc_aided': True,
    }


class TestSCListPolarCode2048_1024_8(VerifySystematicSCListCode):
    code_parameters = {
        'codeword_length': 2048,
        'info_length': 1024,
        'is_systematic': True,
        'list_size': 8,
    }


class TestSCListPolarCode2048_1024_8_crc(VerifySystematicSCListCode):
    code_parameters = {
        'codeword_length': 2048,
        'info_length': 1024,
        'is_systematic': True,
        'list_size': 8,
        'is_crc_aided': True,
    }


class TestSCListPolarCode2048_1024_32(VerifySystematicSCListCode):
    code_parameters = {
        'codeword_length': 2048,
        'info_length': 1024,
        'is_systematic': True,
        'list_size': 32,
    }


class TestSCListPolarCode2048_1024_32_crc(VerifySystematicSCListCode):
    code_parameters = {
        'codeword_length': 2048,
        'info_length': 1024,
        'is_systematic': True,
        'list_size': 32,
        'is_crc_aided': True,
    }
