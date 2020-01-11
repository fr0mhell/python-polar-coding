from unittest import TestCase

import numpy as np

from polar_codes.base import BasicPolarCode


class TestBasicPolarCode(TestCase):
    """Tests for `BasicPolarCode`."""

    @classmethod
    def setUpClass(cls):
        cls.codeword_length = 64
        cls.info_length = 32
        cls.design_snr = 0.0
        cls.non_systematic_code = BasicPolarCode(
            N=cls.codeword_length,
            K=cls.info_length,
            is_systematic=False,
            design_snr=cls.design_snr,
        )

        cls.systematic_code = BasicPolarCode(
            N=cls.codeword_length,
            K=cls.info_length,
            design_snr=cls.design_snr,
        )

        cls.systematic_crc_code = BasicPolarCode(
            N=cls.codeword_length,
            K=cls.info_length,
            design_snr=cls.design_snr,
            is_crc_aided=True,
        )

        # Raw test data
        cls.raw_message = '11011001110110100111101111101010'
        cls.raw_mask = ('00000001000101110001011101010111'
                        '00010101000101110001011101111111')
        cls.raw_mask_crc = ('00010111010101110101011101111111'
                            '01010111011111110111111111111111')
        cls.non_sys_encoded = ('11101011001010000010100011100100'
                               '01000010001010111101101101000010')
        cls.sys_encoded = ('01101001001100111100001101100110'
                           '10011010001111111100111101101010')

    @property
    def message(self):
        """Info message"""
        return np.array([int(m) for m in self.raw_message])

    @property
    def mask(self):
        """Expected mask"""
        return np.array([int(m) for m in self.raw_mask])

    @property
    def mask_crc(self):
        """Expected mask for code with CRC"""
        return np.array([int(m) for m in self.raw_mask_crc])

    @property
    def non_sys_enc_msg(self):
        """Expected non-systematically encoded message"""
        return np.array([int(i) for i in self.non_sys_encoded])

    @property
    def sys_enc_msg(self):
        """Expected systematically encoded message"""
        return np.array([int(i) for i in self.sys_encoded])

    def test_building_polar_mask(self):
        """Test `build_polar_code_mask` method."""
        mask1 = self.non_systematic_code.polar_code_construction()
        mask2 = self.systematic_code.polar_code_construction()
        mask3 = self.systematic_crc_code.polar_code_construction()

        self.assertEqual(np.sum(mask1), self.info_length)
        self.assertTrue(all(mask1 == self.mask))

        self.assertEqual(np.sum(mask2), self.info_length)
        self.assertTrue(all(mask2 == self.mask))

        # CRC code's mask has additional 16 info positions for CRC
        self.assertEqual(np.sum(mask3), self.info_length + 16)
        self.assertTrue(all(mask3 == self.mask_crc))

    def test_precode_and_extract(self):
        """Test `_precode` and `_extract` methods"""
        precoded = self.systematic_code._precode(self.message)
        self.assertEqual(precoded.size, self.codeword_length)

        extracted = self.systematic_code._extract(precoded)
        self.assertEqual(extracted.size, self.info_length)

        self.assertTrue(all(extracted == self.message))

    def test_non_systematic_encode(self):
        """Test `encode` method for non-systematic code."""
        encoded = self.non_systematic_code.encode(self.message)
        self.assertTrue(all(encoded == self.non_sys_enc_msg))

    def test_systematic_encode(self):
        """Test `encode` method for systematic code."""
        encoded = self.systematic_code.encode(self.message)
        self.assertTrue(all(encoded == self.sys_enc_msg))

        extracted = self.systematic_code._extract(encoded)
        self.assertTrue(all(extracted == self.message))

    def test_systematic_encode_with_crc(self):
        """Test for systematic encoding with CRC support"""
        encoded = self.systematic_crc_code.encode(self.message)
        extracted = self.systematic_crc_code._extract(encoded)
        self.assertTrue(all(extracted[:self.info_length] == self.message))
