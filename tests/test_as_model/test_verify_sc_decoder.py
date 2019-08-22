import pprint

from .channels import VerificationChannel
from .test_sc_code import TestSCPolarCode


class TestSystematicSCCode_1024_512(TestSCPolarCode):
    messages = 10000
    channel_class = VerificationChannel
    code_parameters = {
        'codeword_length': 1024,
        'info_length': 512,
        'is_systematic': True
    }

    def _base_test(self, snr_db=0.0, with_noise=True):
        channel = self.channel_class(snr_db, self.K, self.N)

        bit_errors, frame_errors = self._message_transmission_test(
            channel,
            with_noise,
        )

        self.bit_errors_data.update({snr_db: bit_errors})
        self.frame_errors_data.update({snr_db: frame_errors})

        pprint.pprint(self.bit_errors_data)
        pprint.pprint(self.frame_errors_data)

        return bit_errors, frame_errors


class TestSystematicSCCode_1024_256(TestSystematicSCCode_1024_512):
    channel_class = VerificationChannel
    code_parameters = {
        'codeword_length': 1024,
        'info_length': 256,
        'is_systematic': True
    }


class TestSystematicSCCode_1024_768(TestSystematicSCCode_1024_512):
    channel_class = VerificationChannel
    code_parameters = {
        'codeword_length': 1024,
        'info_length': 768,
        'is_systematic': True
    }


class TestSystematicSCCode_2048_512(TestSystematicSCCode_1024_512):
    channel_class = VerificationChannel
    code_parameters = {
        'codeword_length': 2048,
        'info_length': 512,
        'is_systematic': True
    }


class TestSystematicSCCode_2048_1024(TestSystematicSCCode_1024_512):
    channel_class = VerificationChannel
    code_parameters = {
        'codeword_length': 2048,
        'info_length': 1024,
        'is_systematic': True
    }


class TestSystematicSCCode_2048_1536(TestSystematicSCCode_1024_512):
    channel_class = VerificationChannel
    code_parameters = {
        'codeword_length': 2048,
        'info_length': 1536,
        'is_systematic': True
    }
