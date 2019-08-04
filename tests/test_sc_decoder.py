from unittest import TestCase

import numpy as np

from polar_codes.successive_cancellation.sc_decoder import SCDecoder


class TestSCDecoder(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.message = np.array([
            -2.7273,
            -8.7327,
            0.1087,
            1.6463,
            0.0506,
            -0.0552,
            -1.5304,
            -2.1233
        ])
        cls.mask = np.array([0, 1, 0, 1, 0, 1, 1, 1, ], dtype=np.int8)
        cls.decoded = np.array([0, 0, 0, 0, 1, 0, 1, 1], dtype=np.int8)
        cls.decoder = SCDecoder(cls.message, cls.mask, is_systematic=False)

        cls.expected_llrs = [
            [
                np.array([2.7273, 0.1087, -0.0506, 1.5304]),
                np.array([0.1087, -0.0506]),
                np.array([-0.0506]),
            ],
            [
                np.array([2.7273, 0.1087, -0.0506, 1.5304]),
                np.array([0.1087, -0.0506]),
                np.array([0.0581]),
            ],
            [
                np.array([2.7273, 0.1087, -0.0506, 1.5304]),
                np.array([2.836, 1.4798]),
                np.array([1.4798]),
            ],
            [
                np.array([2.7273, 0.1087, -0.0506, 1.5304]),
                np.array([2.836, 1.4798]),
                np.array([4.3158]),
            ],
            [
                np.array([-11.46, 1.755, -0.0046, -3.6537]),
                np.array([-1.755, 0.0046]),
                np.array([-0.0046]),
            ],
            [
                np.array([-11.46, 1.755, -0.0046, -3.6537]),
                np.array([-1.755, 0.0046]),
                np.array([1.7596]),
            ],
            [
                np.array([-11.46, 1.755, -0.0046, -3.6537]),
                np.array([13.215, -3.6583]),
                np.array([-3.6583]),
            ],
            [
                np.array([-11.46, 1.755, -0.0046, -3.6537]),
                np.array([13.215, -3.6583]),
                np.array([-16.8733]),
            ],
        ]
        cls.expected_decoded = [
            np.array([0, 0, 0, 0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0]),
            np.array([0, 1, 0, 0, 0, 0, 0, 0]),
            np.array([0, 1, 0, 0, 0, 0, 0, 0]),
            np.array([0, 1, 0, 1, 0, 0, 0, 0]),
            np.array([0, 1, 0, 1, 0, 0, 0, 1]),
        ]
        cls.expected_bits = [
            [
                np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
                np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
                np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
                np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
            ],
            [
                np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
                np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
                np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
                np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
            ],
            [
                np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
                np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
                np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
                np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
            ],
            [
                np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
                np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
                np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
                np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
            ],
            [
                np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
                np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
                np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
                np.array([0, 1, 0, 0, 0, 0, 0, 0, ]),
            ],
            [
                np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
                np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
                np.array([0, 1, 0, 0, 0, 0, 0, 0, ]),
                np.array([0, 1, 0, 0, 0, 0, 0, 0, ]),
            ],
            [
                np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
                np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
                np.array([0, 1, 0, 0, 0, 0, 0, 0, ]),
                np.array([0, 1, 0, 1, 0, 0, 0, 0, ]),
            ],
            [
                np.array([1, 1, 0, 0, 1, 1, 1, 1, ]),
                np.array([0, 1, 0, 0, 0, 1, 0, 1, ]),
                np.array([0, 1, 0, 0, 0, 0, 0, 1, ]),
                np.array([0, 1, 0, 1, 0, 0, 0, 1, ]),
            ],
        ]

    def test_initial_params(self):
        self.assertEqual(self.decoder.n, 3)

        start = self.decoder.N
        for stage in self.decoder.intermediate_llr:
            start = start // 2
            self.assertEqual(stage.size, start)

    def _decoding_step(self, step):
        """Single step of decoding process."""
        self.decoder.set_decoder_state(step)

        # Check intermediate LLRs computation
        expected_llr = self.expected_llrs[step]
        self.decoder.compute_intermediate_llr()
        for i in range(self.decoder.n):
            np.testing.assert_array_almost_equal(
                self.decoder.intermediate_llr[i],
                expected_llr[i]
            )

        # Check decoding result
        decoded = self.decoder.make_decision()
        self.decoder.compute_intermediate_bits(decoded)

        # Check intermediate bits computation
        expected_bits = self.expected_bits[step]
        np.testing.assert_array_equal(
            self.decoder.result,
            self.expected_decoded[step]
        )

        for i in range(self.decoder.n):
            np.testing.assert_equal(
                self.decoder.intermediate_bits[i],
                expected_bits[i]
            )

        self.decoder.set_next_decoding_position()

    def test_decoding_steps(self):
        """Test SC decoding process step-by-step."""
        for i in range(self.message.size):
            self._decoding_step(i)

    def test_decoding_step_by_step(self):
        """Test `decoder_step` method."""
        for i in range(self.message.size):
            self.decoder.decoder_step(i)

        np.testing.assert_array_equal(self.decoder.result, self.decoded)
