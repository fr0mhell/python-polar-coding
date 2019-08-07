from unittest import TestCase

import numpy as np

from polar_codes.base.functions import compute_encoding_step


class TestBaseFunctions(TestCase):
    """Tests for polar coding basic functions."""

    def setUp(self):
        self.N = 8
        self.n = 3

    def test_compute_encoding_step(self):
        """Test `compute_encoding_step` function."""
        level = self.n
        source = np.ones(self.N, dtype=np.int8)
        result = source

        level -= 1
        result = compute_encoding_step(level, self.n, source, result)
        np.testing.assert_array_equal(
            result,
            np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
        )

        level -= 1
        source = result
        result = compute_encoding_step(level, self.n, source, result)
        np.testing.assert_array_equal(
            result,
            np.array([0, 0, 0, 1, 0, 0, 0, 1], dtype=np.int8)
        )

        level -= 1
        source = result
        result = compute_encoding_step(level, self.n, source, result)
        np.testing.assert_array_equal(
            result,
            np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.int8)
        )
