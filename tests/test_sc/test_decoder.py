from unittest import TestCase

import numpy as np

from python_polar_coding.polar_codes.sc import SCDecoder


class TestSCDecoder(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.received_llr = np.array([
            -2.7273, -8.7327,  0.1087,  1.6463,
             0.0506, -0.0552, -1.5304, -2.1233,
        ])
        cls.mask = np.array([0, 1, 0, 1, 0, 1, 1, 1, ], dtype=np.int8)
        cls.steps = cls.mask.size
        cls.decoder = SCDecoder(mask=cls.mask, is_systematic=False, n=3)

        cls.expected_llrs = [
            [
                cls.received_llr,
                np.array([-0.0506, 0.0552, -0.1087, -1.6463]),
                np.array([0.0506, -0.0552]),
                np.array([-0.0506]),
            ],
            [
                cls.received_llr,
                np.array([-0.0506, 0.0552, -0.1087, -1.6463]),
                np.array([0.0506, -0.0552]),
                np.array([-0.0046]),
            ],
            [
                cls.received_llr,
                np.array([-0.0506, 0.0552, -0.1087, -1.6463]),
                np.array([-0.0581, -1.7015]),
                np.array([0.0581]),
            ],
            [
                cls.received_llr,
                np.array([-0.0506, 0.0552, -0.1087, -1.6463]),
                np.array([-0.0581, -1.7015]),
                np.array([-1.7596]),
            ],
            [
                cls.received_llr,
                np.array([-2.6767, -8.7879, -1.6391, -3.7696]),
                np.array([1.6391, 3.7696]),
                np.array([1.6391]),
            ],
            [
                cls.received_llr,
                np.array([-2.6767, -8.7879, -1.6391, -3.7696]),
                np.array([1.6391, 3.7696]),
                np.array([5.4087]),
            ],
            [
                cls.received_llr,
                np.array([-2.6767, -8.7879, -1.6391, -3.7696]),
                np.array([-4.3158, -12.5575]),
                np.array([4.3158]),
            ],
            [
                cls.received_llr,
                np.array([-2.6767, -8.7879, -1.6391, -3.7696]),
                np.array([-4.3158, -12.5575]),
                np.array([-16.8733]),
            ],
        ]
        cls.expected_decoded = np.array([0, 1, 0, 1, 0, 0, 0, 1])
        cls.expected_bits = [
            [
                np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
                np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
                np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
                np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
            ],
            [
                np.array([1, 1, 0, 0, 0, 0, 0, 0, ]),
                np.array([1, 1, 0, 0, 0, 0, 0, 0, ]),
                np.array([1, 1, 0, 0, 0, 0, 0, 0, ]),
                np.array([0, 1, 0, 0, 0, 0, 0, 0, ]),
            ],
            [
                np.array([1, 1, 0, 0, 0, 0, 0, 0, ]),
                np.array([1, 1, 0, 0, 0, 0, 0, 0, ]),
                np.array([1, 1, 0, 0, 0, 0, 0, 0, ]),
                np.array([0, 1, 0, 0, 0, 0, 0, 0, ]),
            ],
            [
                np.array([0, 0, 1, 1, 0, 0, 0, 0, ]),
                np.array([0, 0, 1, 1, 0, 0, 0, 0, ]),
                np.array([1, 1, 1, 1, 0, 0, 0, 0, ]),
                np.array([0, 1, 0, 1, 0, 0, 0, 0, ]),
            ],
            [
                np.array([0, 0, 1, 1, 0, 0, 0, 0, ]),
                np.array([0, 0, 1, 1, 0, 0, 0, 0, ]),
                np.array([1, 1, 1, 1, 0, 0, 0, 0, ]),
                np.array([0, 1, 0, 1, 0, 0, 0, 0, ]),
            ],
            [
                np.array([0, 0, 1, 1, 0, 0, 0, 0, ]),
                np.array([0, 0, 1, 1, 0, 0, 0, 0, ]),
                np.array([1, 1, 1, 1, 0, 0, 0, 0, ]),
                np.array([0, 1, 0, 1, 0, 0, 0, 0, ]),
            ],
            [
                np.array([0, 0, 1, 1, 0, 0, 0, 0, ]),
                np.array([0, 0, 1, 1, 0, 0, 0, 0, ]),
                np.array([1, 1, 1, 1, 0, 0, 0, 0, ]),
                np.array([0, 1, 0, 1, 0, 0, 0, 0, ]),
            ],
            [
                np.array([1, 1, 0, 0, 1, 1, 1, 1, ]),
                np.array([0, 0, 1, 1, 1, 1, 1, 1, ]),
                np.array([1, 1, 1, 1, 0, 0, 1, 1, ]),
                np.array([0, 1, 0, 1, 0, 0, 0, 1, ]),
            ],
        ]

    def _decoding_step(self, position):
        """Single step of decoding process."""
        self.decoder._set_decoder_state(position)

        # Check intermediate LLRs computation
        expected_llr = self.expected_llrs[position]
        self.decoder._compute_intermediate_alpha(position)
        for i in range(self.decoder.n + 1):
            np.testing.assert_array_almost_equal(
                self.decoder.intermediate_llr[i],
                expected_llr[i]
            )

        # Check decoding result
        self.decoder._compute_beta(position)
        decoded = self.decoder._current_decision
        self.assertEqual(decoded, self.expected_decoded[position])

        # Check intermediate bits computation
        self.decoder._compute_intermediate_beta(position)
        expected_bits = self.expected_bits[position]
        for i in range(self.decoder.n + 1):
            np.testing.assert_array_almost_equal(
                self.decoder.intermediate_bits[i],
                expected_bits[i]
            )

        self.decoder._update_decoder_state()

    def test_decoding_steps(self):
        """Test SC decoding process step-by-step."""
        self.decoder._set_initial_state(self.received_llr)
        for i in range(self.steps):
            self._decoding_step(i)
