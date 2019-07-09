from unittest import TestCase
from polar_codes import SCPolarCode
from .datasets import sc_decoder_dataset_non_systematic as non_systematic_ds
from .datasets import sc_decoder_dataset_systematic as systematic_ds
from .mixins import BasicPolarDecoderTestMixin


class SCCodeNonSystematic(BasicPolarDecoderTestMixin, TestCase):
    """Tests for non-systematic SC decoder."""
    input_dataset = non_systematic_ds.LLR_INPUT
    output_dataset = non_systematic_ds.BIT_OUTPUT
    code_class = SCPolarCode

    @property
    def is_systematic(self):
        return False

    def test_8_5_code(self):
        self.common_params.update({
            'codeword_length': 8,
            'info_length': 5,
        })
        code = self.code_class(**self.common_params)
        input_vector = [
            3.9226,
            0.2339,
            -1.8453,
            1.2222,
            -4.6747,
            5.1177,
            -1.6383,
            2.4204,
        ]
        output_vector = [1, 0, 0, 0, 0]

        result = code.decode(input_vector)
        self.assertListEqual(list(result), output_vector)

    def test_64_32_code(self):
        self.common_params.update({
            'codeword_length': 64,
            'info_length': 32,
        })
        self._check_decoder_on_dataset()

    def test_128_96_code(self):
        self.common_params.update({
            'codeword_length': 128,
            'info_length': 96,
        })
        self._check_decoder_on_dataset()

    def test_256_64_code(self):
        self.common_params.update({
            'codeword_length': 256,
            'info_length': 64,
        })
        self._check_decoder_on_dataset()

    def test_512_256_code(self):
        self.common_params.update({
            'codeword_length': 512,
            'info_length': 256,
        })
        self._check_decoder_on_dataset()


class SCCodeSystematic(SCCodeNonSystematic):
    """Tests for systematic SC decoder.

    Same to non-systematic SC decoder but with another dataset.

    """
    input_dataset = systematic_ds.LLR_INPUT
    output_dataset = systematic_ds.BIT_OUTPUT

    @property
    def is_systematic(self):
        return True
