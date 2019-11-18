from polar_codes import RCSCANPolarCode

from .sc import VerifySystematicSCCode


class VerifyRCSCANCode(VerifySystematicSCCode):
    messages = 10000
    codec_class = RCSCANPolarCode

    @classmethod
    def _get_filename(cls):
        N = cls.code_parameters['codeword_length']
        K = cls.code_parameters['info_length']
        I = cls.code_parameters['iterations']
        filename = f'{N}_{K}_I_{I}'
        if cls.code_parameters.get('is_crc_aided'):
            filename += '_crc'
        return f'{filename}.json'


class TestSystematicCode_1024_256_iter_1(VerifyRCSCANCode):
    code_parameters = {
        'codeword_length': 1024,
        'info_length': 256,
        'is_systematic': True,
        'iterations': 1,
    }


class TestSystematicCode_1024_512_iter_1(VerifyRCSCANCode):
    messages = 1000
    code_parameters = {
        'codeword_length': 1024,
        'info_length': 512,
        'is_systematic': True,
        'iterations': 1,
    }


class TestSystematicCode_1024_768_iter_1(VerifyRCSCANCode):
    code_parameters = {
        'codeword_length': 1024,
        'info_length': 768,
        'is_systematic': True,
        'iterations': 1,
    }


class TestSystematicCode_1024_256_iter_2(VerifyRCSCANCode):
    code_parameters = {
        'codeword_length': 1024,
        'info_length': 256,
        'is_systematic': True,
        'iterations': 2,
    }


class TestSystematicCode_1024_512_iter_2(VerifyRCSCANCode):
    code_parameters = {
        'codeword_length': 1024,
        'info_length': 512,
        'is_systematic': True,
        'iterations': 2,
    }


class TestSystematicCode_1024_768_iter_2(VerifyRCSCANCode):
    code_parameters = {
        'codeword_length': 1024,
        'info_length': 768,
        'is_systematic': True,
        'iterations': 2,
    }


class TestSystematicCode_1024_256_iter_4(VerifyRCSCANCode):
    code_parameters = {
        'codeword_length': 1024,
        'info_length': 256,
        'is_systematic': True,
        'iterations': 4,
    }


class TestSystematicCode_1024_512_iter_2_crc(VerifyRCSCANCode):
    code_parameters = {
        'codeword_length': 1024,
        'info_length': 512,
        'is_systematic': True,
        'iterations': 2,
        'is_crc_aided': True,
    }


class TestSystematicCode_1024_512_iter_4(VerifyRCSCANCode):
    code_parameters = {
        'codeword_length': 1024,
        'info_length': 512,
        'is_systematic': True,
        'iterations': 4,
    }


class TestSystematicCode_1024_768_iter_4(VerifyRCSCANCode):
    code_parameters = {
        'codeword_length': 1024,
        'info_length': 768,
        'is_systematic': True,
        'iterations': 4,
    }


# Codes with N = 2048


class TestSystematicCode_2048_512_iter_1(VerifyRCSCANCode):
    code_parameters = {
        'codeword_length': 2048,
        'info_length': 512,
        'is_systematic': True,
        'iterations': 1,
    }


class TestSystematicCode_1024_512_iter_4_crc(VerifyRCSCANCode):
    code_parameters = {
        'codeword_length': 1024,
        'info_length': 512,
        'is_systematic': True,
        'iterations': 4,
        'is_crc_aided': True,
    }


class TestSystematicCode_1024_256_iter_4_crc(VerifyRCSCANCode):
    code_parameters = {
        'codeword_length': 1024,
        'info_length': 256,
        'is_systematic': True,
        'iterations': 4,
        'is_crc_aided': True,
    }


class TestSystematicCode_1024_768_iter_4_crc(VerifyRCSCANCode):
    code_parameters = {
        'codeword_length': 1024,
        'info_length': 768,
        'is_systematic': True,
        'iterations': 4,
        'is_crc_aided': True,
    }


class TestSystematicCode_2048_1024_iter_1(VerifyRCSCANCode):
    code_parameters = {
        'codeword_length': 2048,
        'info_length': 1024,
        'is_systematic': True,
        'iterations': 1,
    }


class TestSystematicCode_2048_1536_iter_1(VerifyRCSCANCode):
    code_parameters = {
        'codeword_length': 2048,
        'info_length': 1536,
        'is_systematic': True,
        'iterations': 1,
    }


class TestSystematicCode_2048_512_iter_2(VerifyRCSCANCode):
    code_parameters = {
        'codeword_length': 2048,
        'info_length': 512,
        'is_systematic': True,
        'iterations': 2,
    }


class TestSystematicCode_2048_1024_iter_2(VerifyRCSCANCode):
    code_parameters = {
        'codeword_length': 2048,
        'info_length': 1024,
        'is_systematic': True,
        'iterations': 2,
    }


class TestSystematicCode_2048_1536_iter_2(VerifyRCSCANCode):
    code_parameters = {
        'codeword_length': 2048,
        'info_length': 1536,
        'is_systematic': True,
        'iterations': 2,
    }


class TestSystematicCode_2048_512_iter_4(VerifyRCSCANCode):
    code_parameters = {
        'codeword_length': 2048,
        'info_length': 512,
        'is_systematic': True,
        'iterations': 4,
    }


class TestSystematicCode_2048_1024_iter_2_crc(VerifyRCSCANCode):
    code_parameters = {
        'codeword_length': 2048,
        'info_length': 1024,
        'is_systematic': True,
        'iterations': 2,
        'is_crc_aided': True,
    }


class TestSystematicCode_2048_1024_iter_4(VerifyRCSCANCode):
    code_parameters = {
        'codeword_length': 2048,
        'info_length': 1024,
        'is_systematic': True,
        'iterations': 4,
    }


class TestSystematicCode_2048_1536_iter_4(VerifyRCSCANCode):
    code_parameters = {
        'codeword_length': 2048,
        'info_length': 1536,
        'is_systematic': True,
        'iterations': 4,
    }


# (8192, 4096) systematic PC


class TestSystematicCode_8192_4096_iter_1(VerifyRCSCANCode):
    code_parameters = {
        'codeword_length': 8192,
        'info_length': 4096,
        'design_snr': 1.4,
        'is_systematic': True,
        'iterations': 1,
    }


class TestSystematicCode_8192_4096_iter_2(VerifyRCSCANCode):
    code_parameters = {
        'codeword_length': 8192,
        'info_length': 4096,
        'design_snr': 1.4,
        'is_systematic': True,
        'iterations': 2,
    }


class TestSystematicCode_8192_4096_iter_4(VerifyRCSCANCode):
    code_parameters = {
        'codeword_length': 8192,
        'info_length': 4096,
        'design_snr': 1.4,
        'is_systematic': True,
        'iterations': 4,
    }


class TestSystematicCode_2048_1024_iter_4_crc(VerifyRCSCANCode):
    code_parameters = {
        'codeword_length': 2048,
        'info_length': 1024,
        'is_systematic': True,
        'iterations': 4,
        'is_crc_aided': True,
    }


class TestSystematicCode_2048_512_iter_4_crc(VerifyRCSCANCode):
    code_parameters = {
        'codeword_length': 2048,
        'info_length': 512,
        'is_systematic': True,
        'iterations': 4,
        'is_crc_aided': True,
    }


class TestSystematicCode_2048_1536_iter_4_crc(VerifyRCSCANCode):
    code_parameters = {
        'codeword_length': 2048,
        'info_length': 1536,
        'is_systematic': True,
        'iterations': 4,
        'is_crc_aided': True,
    }
