from tests.test_as_model.test_verify_fast_ssc_decoder import VerifyFastSSCCode

N = 128
K = 64
dumped_mask = (
    '00000000000000000000000000000001'
    '0000000000000001'
    '00000001'
    '01111111'
    '00000001'
    '01111111'
    '01111111'
    '11111111'
    '01111111111111111111111111111111'
)


class TestSystematicSCCode_128_64(VerifyFastSSCCode):
    code_parameters = {
        'codeword_length': N,
        'info_length': K,
        'is_systematic': True
    }


class TestSystematicSCCode_128_64_manual(VerifyFastSSCCode):
    code_parameters = {
        'codeword_length': N,
        'info_length': K,
        'is_systematic': True,
        'dumped_mask': dumped_mask,
    }

    @classmethod
    def _get_filename(cls):
        return 'manual_' + super()._get_filename()
