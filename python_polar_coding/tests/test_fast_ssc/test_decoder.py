from unittest import TestCase

import numpy as np

from python_polar_coding.polar_codes.fast_ssc import FastSSCDecoder


class TestFastSSCDecoder(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.received_llr = np.array([
            -2.7273,
            -8.7327,
            0.1087,
            1.6463,
            0.0506,
            -0.0552,
            -1.5304,
            -2.1233,
        ])
        cls.length = cls.received_llr.size
        cls.n = 3

    def test_zero_node_decoder(self):
        mask = np.zeros(self.length, dtype=np.int8)
        decoder = FastSSCDecoder(mask=mask, n=self.n)
        self.assertEqual(len(decoder._decoding_tree.leaves), 1)

        decoder.decode(self.received_llr)
        np.testing.assert_equal(
            decoder.result,
            np.zeros(self.length, dtype=np.int8)
        )

    def test_one_node_decoder(self):
        mask = np.ones(self.length, dtype=np.int8)
        decoder = FastSSCDecoder(mask=mask, n=self.n)
        self.assertEqual(len(decoder._decoding_tree.leaves), 1)

        decoder.decode(self.received_llr)
        np.testing.assert_equal(
            decoder.result,
            np.array(self.received_llr < 0, dtype=np.int8)
        )

    def test_spc_node_decoder(self):
        mask = np.array([0, 1, 1, 1, 1, 1, 1, 1], dtype=np.int8)
        decoder = FastSSCDecoder(mask=mask, n=self.n)
        self.assertEqual(len(decoder._decoding_tree.leaves), 1)

        decoder.decode(self.received_llr)
        np.testing.assert_equal(
            decoder.result,
            np.array([1, 1, 0, 0, 1, 1, 1, 1], dtype=np.int8)
        )

    def test_repetition_node_decoder(self):
        mask = np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.int8)
        decoder = FastSSCDecoder(mask=mask, n=self.n)
        self.assertEqual(len(decoder._decoding_tree.leaves), 1)

        decoder.decode(self.received_llr)
        np.testing.assert_equal(
            decoder.result,
            np.ones(self.length, dtype=np.int8)
        )

    def test_repetition_spc_node_decoder(self):
        mask = np.array([0, 0, 0, 1, 0, 1, 1, 1], dtype=np.int8)
        decoder = FastSSCDecoder(mask=mask, n=self.n)
        self.assertEqual(len(decoder._decoding_tree.leaves), 2)

        decoder.decode(self.received_llr)

        # Check nodes
        # Left node
        exp_left_llrs = np.array([-0.0506, 0.0552, -0.1087, -1.6463, ])
        exp_left_bits = np.array([1, 1, 1, 1, ])
        np.testing.assert_array_almost_equal(
            decoder._decoding_tree.leaves[0].alpha,
            exp_left_llrs
        )
        np.testing.assert_equal(
            decoder._decoding_tree.leaves[0].beta,
            exp_left_bits
        )

        # Right node
        exp_right_llrs = np.array([2.7779, 8.6775, -1.6391, -3.7696, ])
        exp_right_bits = np.array([0, 0, 1, 1, ])
        np.testing.assert_array_almost_equal(
            decoder._decoding_tree.leaves[1].alpha,
            exp_right_llrs
        )
        np.testing.assert_equal(
            decoder._decoding_tree.leaves[1].beta,
            exp_right_bits
        )

        # Overall result
        exp_result = np.array([1, 1, 0, 0, 0, 0, 1, 1, ])
        np.testing.assert_equal(decoder.result, exp_result)

    def test_spc_repetition_node_decoder(self):
        mask = np.array([0, 1, 1, 1, 0, 0, 0, 1], dtype=np.int8)
        decoder = FastSSCDecoder(mask=mask, n=self.n)
        self.assertEqual(len(decoder._decoding_tree.leaves), 2)

        decoder.decode(self.received_llr)

        # Check nodes
        # Left node
        exp_left_llrs = np.array([-0.0506, 0.0552, -0.1087, -1.6463, ])
        exp_left_bits = np.array([0, 0, 1, 1, ])
        np.testing.assert_array_almost_equal(
            decoder._decoding_tree.leaves[0].alpha,
            exp_left_llrs
        )
        np.testing.assert_equal(
            decoder._decoding_tree.leaves[0].beta,
            exp_left_bits
        )

        # Right node
        exp_right_llrs = np.array([-2.6767, -8.7879, -1.6391, -3.7696, ])
        exp_right_bits = np.array([1, 1, 1, 1, ])
        np.testing.assert_array_almost_equal(
            decoder._decoding_tree.leaves[1].alpha,
            exp_right_llrs
        )
        np.testing.assert_equal(
            decoder._decoding_tree.leaves[1].beta,
            exp_right_bits
        )

        # Overall result
        exp_result = np.array([1, 1, 0, 0, 1, 1, 1, 1, ])
        np.testing.assert_equal(decoder.result, exp_result)

    def test_complex(self):
        long_msg = np.array([
             0.1139, 1.4662,  2.8427,  0.8675,  1.2576, -1.1791, 0.7535,  2.2528,
            -0.3653, 0.6884, -0.9574, -0.2793, -0.8862, -1.7831, 1.7425, -3.0953,
        ])
        mask = np.array(
            [1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, ], dtype=np.int8)
        sub_codes = [
            np.array([1, 1, ], dtype=np.int8),
            np.array([0, 1, ], dtype=np.int8),
            np.array([0, 0, 0, 1, ], dtype=np.int8),
            np.array([1, ], dtype=np.int8),
            np.array([0, ], dtype=np.int8),
            np.array([1, ], dtype=np.int8),
            np.array([0, ], dtype=np.int8),
            np.array([0, 1, 1, 1, ], dtype=np.int8),
        ]
        decoder = FastSSCDecoder(mask=mask, n=4)

        # Check tree structure
        self.assertEqual(len(decoder._decoding_tree.leaves), len(sub_codes))
        for i, leaf in enumerate(decoder._decoding_tree.leaves):
            np.testing.assert_equal(leaf.mask, sub_codes[i])

        decoder.decode(long_msg)

        # Check nodes
        expected_llr = [
            np.array([-0.1139, 0.2793, ]),
            np.array([-0.8674, 0.9677, ]),
            np.array([-0.7723, 1.8675, -0.2039, -2.5321, ]),
            np.array([-0.2514, ]),
            np.array([0, ]),
            np.array([-1.2404, ]),
            np.array([0, ]),
            np.array([-2.3952, -1.3818, 4.7891, -6.4949, ]),
        ]

        expected_bits = [
            np.array([1, 0, ], dtype=np.int8),
            np.array([0, 0, ], dtype=np.int8),
            np.array([1, 1, 1, 1, ], dtype=np.int8),
            np.array([1, ], dtype=np.int8),
            np.array([0, ], dtype=np.int8),
            np.array([1, ], dtype=np.int8),
            np.array([0, ], dtype=np.int8),
            np.array([1, 0, 0, 1, ], dtype=np.int8),
        ]

        for i, leaf in enumerate(decoder._decoding_tree.leaves):
            np.testing.assert_almost_equal(leaf.alpha, expected_llr[i])
            np.testing.assert_equal(leaf.beta, expected_bits[i])

        # Check result
        np.testing.assert_equal(
            decoder.result,
            np.array([1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, ],
                     dtype=np.int8)
        )
