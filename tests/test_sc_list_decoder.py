"""
left = 0.0506
right = -0.0552
bit = 0
right - (2 * bit - 1) * left
"""
from unittest import TestCase

import numpy as np

from polar_codes.decoders import SCListDecoder


class TestSCDecoder(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.received_llr = np.array([
            -2.7273, -8.7327, 0.1087, 1.6463,
            0.0506, -0.0552, -1.5304, -2.1233,
        ])
        cls.mask = np.array([0, 1, 0, 1, 0, 1, 1, 1, ], dtype=np.int8)
        cls.steps = cls.mask.size
        cls.decoder = SCListDecoder(
            cls.mask,
            is_systematic=False,
            list_size=4,
        )

        cls.expected_llrs = [
            # Step 0
            [
                [
                    cls.received_llr,
                    np.array([-0.0506, 0.0552, -0.1087, -1.6463]),
                    np.array([0.0506, -0.0552]),
                    np.array([-0.0506]),
                ],
            ],
            # Step 1
            [
                [
                    cls.received_llr,
                    np.array([-0.0506, 0.0552, -0.1087, -1.6463]),
                    np.array([0.0506, -0.0552]),
                    np.array([-0.0046]),
                ],
                [
                    cls.received_llr,
                    np.array([-0.0506, 0.0552, -0.1087, -1.6463]),
                    np.array([0.0506, -0.0552]),
                    np.array([-0.0046]),
                ],
            ],
            # Step 2
            [
                [
                    cls.received_llr,
                    np.array([-0.0506, 0.0552, -0.1087, -1.6463]),
                    np.array([-0.0581, -1.7015]),
                    np.array([0.0581]),
                ],
                [
                    cls.received_llr,
                    np.array([-0.0506, 0.0552, -0.1087, -1.6463]),
                    np.array([-0.1593, -1.5911]),
                    np.array([0.1593]),
                ],
            ],
            # Step 3
            [
                [
                    cls.received_llr,
                    np.array([-0.0506, 0.0552, -0.1087, -1.6463]),
                    np.array([-0.0581, -1.7015]),
                    np.array([-1.7596]),
                ],
                [
                    cls.received_llr,
                    np.array([-0.0506, 0.0552, -0.1087, -1.6463]),
                    np.array([-0.1593, -1.5911]),
                    np.array([-1.7504]),
                ],
                [
                    cls.received_llr,
                    np.array([-0.0506, 0.0552, -0.1087, -1.6463]),
                    np.array([-0.1593, -1.5911]),
                    np.array([-1.7504]),
                ],
                [
                    cls.received_llr,
                    np.array([-0.0506, 0.0552, -0.1087, -1.6463]),
                    np.array([-0.0581, -1.7015]),
                    np.array([-1.7596]),
                ],
            ],
            # Step 4
            [
                [
                    cls.received_llr,
                    np.array([-2.6767, -8.7879, -1.6391, -3.7696]),
                    np.array([1.6391, 3.7696]),
                    np.array([1.6391]),
                ],
                [
                    cls.received_llr,
                    np.array([2.7779, 8.6775, -1.6391, -3.7696]),
                    np.array([-1.6391, -3.7696]),
                    np.array([1.6391]),
                ],
                [
                    cls.received_llr,
                    np.array([-2.6767, 0.0552, -1.4217, -0.477]),
                    np.array([1.4217, -0.0552]),
                    np.array([-0.0552]),
                ],
                [
                    cls.received_llr,
                    np.array([2.7779, -8.7879, -1.4217, -0.477]),
                    np.array([-1.4217, 0.477]),
                    np.array([-0.477]),
                ],
            ],
            # Step 5
            [
                [
                    cls.received_llr,
                    np.array([-2.6767, -8.7879, -1.6391, -3.7696]),
                    np.array([1.6391, 3.7696]),
                    np.array([5.4087]),
                ],
                [
                    cls.received_llr,
                    np.array([2.7779, 8.6775, -1.6391, -3.7696]),
                    np.array([-1.6391, -3.7696]),
                    np.array([-5.4087]),
                ],
                [
                    cls.received_llr,
                    np.array([-2.6767, 0.0552, -1.4217, -0.477]),
                    np.array([1.4217, -0.0552]),
                    np.array([1.3665]),
                ],
                [
                    cls.received_llr,
                    np.array([2.7779, -8.7879, -1.4217, -0.477]),
                    np.array([-1.4217, 0.477]),
                    np.array([-0.9447]),
                ],
            ],
            # Step 6
            [
                [
                    cls.received_llr,
                    np.array([-2.6767, -8.7879, -1.6391, -3.7696]),
                    np.array([-4.3158, -12.5575]),
                    np.array([4.3158]),
                ],
                [
                    cls.received_llr,
                    np.array([2.7779, 8.6775, -1.6391, -3.7696]),
                    np.array([-4.417, -12.4471]),
                    np.array([4.417]),
                ],
                [
                    cls.received_llr,
                    np.array([-2.6767, 0.0552, -1.4217, -0.477]),
                    np.array([-4.0984, -0.4218]),
                    np.array([0.4218]),
                ],
                [
                    cls.received_llr,
                    np.array([2.7779, -8.7879, -1.4217, -0.477]),
                    np.array([-4.1996, 8.3109]),
                    np.array([-4.1996]),
                ],
            ],
            # Step 7
            [
                [
                    cls.received_llr,
                    np.array([-2.6767, -8.7879, -1.6391, -3.7696]),
                    np.array([-4.3158, -12.5575]),
                    np.array([-16.8733]),
                ],
                [
                    cls.received_llr,
                    np.array([2.7779, 8.6775, -1.6391, -3.7696]),
                    np.array([-4.417, -12.4471]),
                    np.array([-16.8641]),
                ],
                [
                    cls.received_llr,
                    np.array([-2.6767, 0.0552, -1.4217, -0.477]),
                    np.array([-4.0984, -0.4218]),
                    np.array([-4.5202]),
                ],
                [
                    cls.received_llr,
                    np.array([2.7779, -8.7879, -1.4217, -0.477]),
                    np.array([-4.1996, 8.3109]),
                    np.array([12.5105]),
                ],
            ],
        ]

        cls.expected_metrics = [
            # Step 0
            [1, ],
            # Step 1
            [0.50115, 0.49885],
            # Step 2
            [0.50115, 0.49885],
            # Step 3
            [0.42756, 0.42502, 0.07383, 0.07359],
            # Step 4
            [0.42756, 0.42502, 0.07383, 0.07359],
            # Step 5
            [0.42565, 0.42313, 0.05883, 0.05299],
            # Step 6
            [0.42004, 0.41808, 0.03553, 0.05221],
            # Step 7
            [0.42004, 0.41808, 0.03515, 0.05221],
        ]

        cls.expected_bits = [
            # Step 0
            [
                [
                    np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
                    np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
                    np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
                    np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
                ],
            ],
            # Step 1
            [
                [
                    np.array([1, 1, 0, 0, 0, 0, 0, 0, ]),
                    np.array([1, 1, 0, 0, 0, 0, 0, 0, ]),
                    np.array([1, 1, 0, 0, 0, 0, 0, 0, ]),
                    np.array([0, 1, 0, 0, 0, 0, 0, 0, ]),
                ],
                [
                    np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
                    np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
                    np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
                    np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
                ],
            ],
            # Step 2
            [
                [
                    np.array([1, 1, 0, 0, 0, 0, 0, 0, ]),
                    np.array([1, 1, 0, 0, 0, 0, 0, 0, ]),
                    np.array([1, 1, 0, 0, 0, 0, 0, 0, ]),
                    np.array([0, 1, 0, 0, 0, 0, 0, 0, ]),
                ],
                [
                    np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
                    np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
                    np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
                    np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
                ],
            ],
            # Step 3
            [
                [
                    np.array([0, 0, 1, 1, 0, 0, 0, 0, ]),
                    np.array([0, 0, 1, 1, 0, 0, 0, 0, ]),
                    np.array([1, 1, 1, 1, 0, 0, 0, 0, ]),
                    np.array([0, 1, 0, 1, 0, 0, 0, 0, ]),
                ],
                [
                    np.array([1, 1, 1, 1, 0, 0, 0, 0, ]),
                    np.array([1, 1, 1, 1, 0, 0, 0, 0, ]),
                    np.array([0, 0, 1, 1, 0, 0, 0, 0, ]),
                    np.array([0, 0, 0, 1, 0, 0, 0, 0, ]),
                ],
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
            ],
            # Step 4
            [
                [
                    np.array([0, 0, 1, 1, 0, 0, 0, 0, ]),
                    np.array([0, 0, 1, 1, 0, 0, 0, 0, ]),
                    np.array([1, 1, 1, 1, 0, 0, 0, 0, ]),
                    np.array([0, 1, 0, 1, 0, 0, 0, 0, ]),
                ],
                [
                    np.array([1, 1, 1, 1, 0, 0, 0, 0, ]),
                    np.array([1, 1, 1, 1, 0, 0, 0, 0, ]),
                    np.array([0, 0, 1, 1, 0, 0, 0, 0, ]),
                    np.array([0, 0, 0, 1, 0, 0, 0, 0, ]),
                ],
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
            ],
            # Step 5
            [
                [
                    np.array([0, 0, 1, 1, 0, 0, 0, 0, ]),
                    np.array([0, 0, 1, 1, 0, 0, 0, 0, ]),
                    np.array([1, 1, 1, 1, 0, 0, 0, 0, ]),
                    np.array([0, 1, 0, 1, 0, 0, 0, 0, ]),
                ],
                [
                    np.array([0, 0, 1, 1, 1, 1, 0, 0, ]),
                    np.array([1, 1, 1, 1, 1, 1, 0, 0, ]),
                    np.array([0, 0, 1, 1, 1, 1, 0, 0, ]),
                    np.array([0, 0, 0, 1, 0, 1, 0, 0, ]),
                ],
                [
                    np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
                    np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
                    np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
                    np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
                ],
                [
                    np.array([0, 0, 0, 0, 1, 1, 0, 0, ]),
                    np.array([1, 1, 0, 0, 1, 1, 0, 0, ]),
                    np.array([1, 1, 0, 0, 1, 1, 0, 0, ]),
                    np.array([0, 1, 0, 0, 0, 1, 0, 0, ]),
                ],
            ],
            # Step 6
            [
                [
                    np.array([0, 0, 1, 1, 0, 0, 0, 0, ]),
                    np.array([0, 0, 1, 1, 0, 0, 0, 0, ]),
                    np.array([1, 1, 1, 1, 0, 0, 0, 0, ]),
                    np.array([0, 1, 0, 1, 0, 0, 0, 0, ]),
                ],
                [
                    np.array([0, 0, 1, 1, 1, 1, 0, 0, ]),
                    np.array([1, 1, 1, 1, 1, 1, 0, 0, ]),
                    np.array([0, 0, 1, 1, 1, 1, 0, 0, ]),
                    np.array([0, 0, 0, 1, 0, 1, 0, 0, ]),
                ],
                [
                    np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
                    np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
                    np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
                    np.array([0, 0, 0, 0, 0, 0, 0, 0, ]),
                ],
                [
                    np.array([1, 0, 1, 0, 0, 1, 1, 0, ]),
                    np.array([1, 1, 0, 0, 0, 1, 1, 0, ]),
                    np.array([1, 1, 0, 0, 1, 1, 1, 0, ]),
                    np.array([0, 1, 0, 0, 0, 1, 1, 0, ]),
                ],
            ],
            # Step 7
            [
                [
                    np.array([1, 1, 0, 0, 1, 1, 1, 1, ]),
                    np.array([0, 0, 1, 1, 1, 1, 1, 1, ]),
                    np.array([1, 1, 1, 1, 0, 0, 1, 1, ]),
                    np.array([0, 1, 0, 1, 0, 0, 0, 1, ]),
                ],
                [
                    np.array([1, 1, 0, 0, 0, 0, 1, 1, ]),
                    np.array([1, 1, 1, 1, 0, 0, 1, 1, ]),
                    np.array([0, 0, 1, 1, 1, 1, 1, 1, ]),
                    np.array([0, 0, 0, 1, 0, 1, 0, 1, ]),
                ],
                [
                    np.array([1, 1, 1, 1, 1, 1, 1, 1, ]),
                    np.array([0, 0, 0, 0, 1, 1, 1, 1, ]),
                    np.array([0, 0, 0, 0, 0, 0, 1, 1, ]),
                    np.array([0, 0, 0, 0, 0, 0, 0, 1, ]),
                ],
                [
                    np.array([1, 0, 1, 0, 0, 1, 1, 0, ]),
                    np.array([1, 1, 0, 0, 0, 1, 1, 0, ]),
                    np.array([1, 1, 0, 0, 1, 1, 1, 0, ]),
                    np.array([0, 1, 0, 0, 0, 1, 1, 0, ]),
                ],
            ],
        ]

    def _decoding_step(self, position):
        """Single step of decoding process."""
        self.decoder(position)
        for i, path in enumerate(self.decoder.paths):

            expected_llr = self.expected_llrs[position]
            for j in range(path.n + 1):
                np.testing.assert_array_almost_equal(
                    path.intermediate_llr[j],
                    expected_llr[i][j]
                )

            expected_bits = self.expected_bits[position]
            for j in range(path.n + 1):
                np.testing.assert_array_equal(
                    path.intermediate_bits[j],
                    expected_bits[i][j]
                )

            expected_metrics = self.expected_metrics[position]
            np.testing.assert_almost_equal(
                path.path_metric,
                expected_metrics[i],
                decimal=4
            )

    def test_decoding_steps(self):
        """Test SC decoding process step-by-step."""
        self.decoder.initialize(self.received_llr)
        for i in range(self.steps):
            self._decoding_step(i)
