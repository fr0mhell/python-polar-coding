from unittest import TestCase

import numpy as np

from tree_builder.tree_builder import FastSSCTreeBuilder


class TestFastSSCTreeBuilder(TestCase):
    """Tests for `FastSSCTreeBuilder`."""

    # @classmethod
    def setUp(cls):  # def setUpClass(cls):
        cls.tree_builder = FastSSCTreeBuilder(
            codeword_length=16,
            info_length=8,
            dumped_mask='0111'
                        '0011'
                        '0001'
                        '0101',
            code_min_size=4
        )
        cls.tree_builder.code.channel_estimates = np.repeat(0.1, 16)
        cls.sub_codes = cls.tree_builder._get_sub_codes()
        cls.fast_ssc, cls.other = cls.tree_builder._group_sub_codes_by_type(
            cls.sub_codes
        )

        # Expected results
        cls.expected_masks = [
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
            [0, 1, 0, 1],
        ]
        cls.expected_fast_ssc_masks = [[0, 0, 0, 1], [0, 1, 1, 1], ]
        cls.expected_other_masks = [[0, 0, 1, 1], [0, 1, 0, 1], ]

        cls.tree_builder_2 = FastSSCTreeBuilder(
            codeword_length=32,
            info_length=24,
            dumped_mask='01111111'
                        '01111111'
                        '01111111'
                        '01000011',
            code_min_size=8
        )
        cls.tree_builder_2.code.channel_estimates = np.repeat(0.1, 32)
        sub_codes = cls.tree_builder_2._get_sub_codes()
        cls.fast_ssc_2, cls.other_2 = \
            cls.tree_builder_2._group_sub_codes_by_type(sub_codes)

    def test_get_sub_codes(self):
        """Test `_get_sub_codes` method."""
        for i, s in enumerate(self.sub_codes):
            self.assertListEqual(s['mask'], self.expected_masks[i])

    def test_group_sub_codes_by_type(self):
        """Test `_group_sub_codes_by_type` method."""
        for i, f in enumerate(self.fast_ssc):
            self.assertIn(f['mask'], self.expected_fast_ssc_masks)

        for i, o in enumerate(self.other):
            self.assertIn(o['mask'], self.expected_other_masks)

    def test_check_other_sub_codes_can_be_rebuilt(self):
        """Test `_check_other_sub_codes_can_be_rebuilt` method."""
        self.assertTrue(
            self.tree_builder._check_other_sub_codes_can_be_rebuilt(self.other)
        )

    def test_check_other_sub_codes_can_be_rebuilt_fails(self):
        """Test `_check_other_sub_codes_can_be_rebuilt` method.

        Method returns False if cannot be rebuilt.

        """
        self.assertFalse(
            self.tree_builder_2._check_other_sub_codes_can_be_rebuilt(self.other_2)  # noqa
        )

    def test_update_other_sub_codes(self):
        """Test `_update_other_sub_codes` method."""
        exp_fast_ssc = [[0, 1, 1, 1, 1, 1, 1, 1], ]
        exp_other = [
            [0, 1, 0, 0, 0, 0, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1],
        ]

        upd_fast_ssc, upd_other = self.tree_builder_2._update_other_sub_codes(
            self.fast_ssc_2, self.other_2,
        )

        for i, f in enumerate(upd_fast_ssc):
            self.assertListEqual(f['mask'], exp_fast_ssc[i])

        for i, o in enumerate(upd_other):
            self.assertListEqual(o['mask'], exp_other[i])

    # TODO: Add tests for code with sub-codes of different size
