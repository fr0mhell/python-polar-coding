from unittest import TestCase

import numpy as np

from python_polar_coding.polar_codes.rc_scan import INFINITY, RCSCANDecoder


class TestRCSCANDecoder(TestCase):
    # @classmethod
    def setUp(cls):
        cls.received_llr = np.array([
            -2.7273, -8.7327,  0.1087,  1.6463,
             0.0506, -0.0552, -1.5304, -2.1233,
        ])
        cls.n = 3
        cls.length = cls.received_llr.size

    def test_zero_node_decoder(self):
        mask = np.zeros(self.length, dtype=np.int8)
        decoder = RCSCANDecoder(mask=mask, n=self.n)
        decoder.decode(self.received_llr)

        self.assertEqual(len(decoder._decoding_tree.leaves), 1)
        np.testing.assert_equal(
            decoder.root.beta,
            np.ones(self.length, dtype=np.double) * INFINITY
        )

    def test_one_node_decoder(self):
        mask = np.ones(self.length, dtype=np.int8)
        decoder = RCSCANDecoder(mask=mask, n=self.n)
        decoder.decode(self.received_llr)

        self.assertEqual(len(decoder._decoding_tree.leaves), 1)
        np.testing.assert_equal(
            decoder.root.beta,
            np.zeros(self.length)
        )


class TestRCSCANDecoderComplex(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mask = np.array(
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, ],
            dtype=np.int8
        )
        cls.n = 4
        cls.sub_codes = [
            np.array([0, 0, 0, 0, ], dtype=np.int8),
            np.array([0, ], dtype=np.int8),
            np.array([1, ], dtype=np.int8),
            np.array([1, 1, ], dtype=np.int8),
            np.array([0, ], dtype=np.int8),
            np.array([1, ], dtype=np.int8),
            np.array([1, 1, ], dtype=np.int8),
            np.array([1, 1, 1, 1, ], dtype=np.int8),
        ]

    def _get_decoder(self):
        return RCSCANDecoder(mask=self.mask, n=self.n)

    def test_sub_codes(self):
        """Check sub-codes built correctly."""
        decoder = self._get_decoder()

        self.assertEqual(
            len(decoder._decoding_tree.leaves),
            len(self.sub_codes)
        )

        for i, leaf in enumerate(decoder._decoding_tree.leaves):
            np.testing.assert_equal(leaf.mask, self.sub_codes[i])

    def test_no_noise(self):
        decoder = self._get_decoder()
        llr = np.array([
            -2, -2, 2, -2, -2, 2, -2,  2,
             2, -2, 2, -2,  2, 2, -2, -2,
        ])

        decoder.decode(llr)

        # Check result
        expected_result_beta = np.array(
            [-2, -2, 2,  6, -2, 2, -2,  2,
              2, -2, 2, -2,  2, 2, -2, -2, ],
            dtype=np.float64
        )
        np.testing.assert_equal(
            decoder._compute_result_beta(),
            expected_result_beta
        )

        expected_result = np.array(
            [1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, ],
            dtype=np.int8
        )
        np.testing.assert_equal(decoder.result, expected_result)

    def test_with_some_noise(self):
        decoder = self._get_decoder()
        llr = np.array([
            -1.9, -1.7, 2.6, -1.7, -1.1,  2.6, -1.3,  2.4,
             2.2, -1.8, 2.1, -1.9,  2.3,  2.2, -1.5, -1.2,
        ])

        decoder.decode(llr)

        # Check result
        # Iteration 1
        expected_result_beta = np.array(
            [-0.6, -1.7, 0.8,  4.2, -1.4, 1.2, -1.6,  1.3,
              0.8, -1.9, 1,   -0.8,  1.3, 1.4, -1.5, -1.5, ],
            dtype=np.float64
        )
        np.testing.assert_almost_equal(
            decoder._compute_result_beta(),
            expected_result_beta
        )

        expected_result = np.array(
            [1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, ],
            dtype=np.int8
        )
        np.testing.assert_equal(decoder.result, expected_result)
