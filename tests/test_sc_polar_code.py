from unittest import TestCase
from polar_codes import SCPolarCode
from .datasets.sc_decoder_dataset import LLR_INPUT, BIT_OUTPUT
from .mixins import BasicPolarDecoderTestMixin


class SCCodeNonSystematic(BasicPolarDecoderTestMixin, TestCase):
    """Tests for non-systematic SC decoder."""
    input_dataset = LLR_INPUT
    output_dataset = BIT_OUTPUT
    code_class = SCPolarCode

    @property
    def is_systematic(self):
        return False

    def test_64_32_code(self):
        self.common_params.update({
            'codeword_length': 64,
            'info_length': 32,
        })
        self.basic_test_case()

    def test_128_96_code(self):
        self.common_params.update({
            'codeword_length': 128,
            'info_length': 96,
        })
        self.basic_test_case()

    def test_256_64_code(self):
        self.common_params.update({
            'codeword_length': 256,
            'info_length': 64,
        })
        self.basic_test_case()

    def test_512_256_code(self):
        self.common_params.update({
            'codeword_length': 512,
            'info_length': 256,
        })
        self.basic_test_case()


class SCCodeSystematic(SCCodeNonSystematic):

    @property
    def is_systematic(self):
        return True
