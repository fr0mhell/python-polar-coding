from unittest import TestCase

import numpy as np

from python_polar_coding.polar_codes.sc_list import SCListDecoder


class TestSCListDecoder(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.received_llr = np.array([
            -2.7273, -8.7327, 0.1087, 1.6463,
            0.0506, -0.0552, -1.5304, -2.1233,
        ])
        cls.mask = np.array([0, 1, 0, 1, 0, 1, 1, 1, ], dtype=np.int8)
        cls.decoder = SCListDecoder(
            n=3,
            mask=cls.mask,
            is_systematic=False,
            L=4,
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
                    np.array([-2.6767, -8.7879, -1.4217, -0.477]),
                    np.array([1.4217, 0.477]),
                    np.array([0.477]),
                ],
                [
                    cls.received_llr,
                    np.array([2.7779, 8.6775, -1.4217, -0.477]),
                    np.array([-1.4217, -0.477]),
                    np.array([0.477]),
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
                    np.array([-2.6767, -8.7879, -1.4217, -0.477]),
                    np.array([1.4217, 0.477]),
                    np.array([1.8987]),
                ],
                [
                    cls.received_llr,
                    np.array([2.7779,  8.6775, -1.4217, -0.477]),
                    np.array([-1.4217, -0.477]),
                    np.array([-1.8987]),
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
                    np.array([-2.6767, -8.7879, -1.4217, -0.477]),
                    np.array([-4.0984, -9.2649]),
                    np.array([4.0984]),
                ],
                [
                    cls.received_llr,
                    np.array([2.7779,  8.6775, -1.4217, -0.477]),
                    np.array([-4.1996, -9.1545]),
                    np.array([4.1996]),
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
                    np.array([-2.6767, -8.7879, -1.4217, -0.477]),
                    np.array([-4.0984, -9.2649]),
                    np.array([-13.3633]),
                ],
                [
                    cls.received_llr,
                    np.array([2.7779,  8.6775, -1.4217, -0.477]),
                    np.array([-4.1996, -9.1545]),
                    np.array([-13.3541]),
                ],
            ],
        ]

        cls.expected_metrics = [
            # Step 0
            [-0.0506, ],
            # Step 1
            [-0.0506, -0.0552],
            # Step 2
            [-0.0506, -0.0552],
            # Step 3
            [-0.0506, -0.0552, -1.8056, -1.8102],
            # Step 4
            [-0.0506, -0.0552, -1.8056, -1.8102],
            # Step 5
            [-0.0506, -0.0552, -1.8056, -1.8102],
            # Step 6
            [-0.0506, -0.0552, -1.8056, -1.8102],
            # Step 7
            [-0.0506, -0.0552, -1.8056, -1.8102],
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
                    np.array([0, 0, 0, 0, 1, 1, 0, 0, ]),
                    np.array([1, 1, 0, 0, 1, 1, 0, 0, ]),
                    np.array([1, 1, 0, 0, 1, 1, 0, 0, ]),
                    np.array([0, 1, 0, 0, 0, 1, 0, 0, ]),
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
                    np.array([1, 1, 1, 1, 0, 0, 1, 1, ]),
                    np.array([1, 1, 0, 0, 0, 0, 1, 1, ]),
                    np.array([1, 1, 0, 0, 1, 1, 1, 1, ]),
                    np.array([0, 1, 0, 0, 0, 1, 0, 1, ]),
                ],
            ],
        ]

    @property
    def K(self):
        return np.sum(self.mask)

    @property
    def N(self):
        return self.mask.size

    def _decoding_step(self, position):
        """Single step of decoding process."""
        self.decoder._decode_position(position)
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
                path._path_metric,
                expected_metrics[i],
                decimal=4
            )

    def test_decoding_steps(self):
        """Test SC list decoding process step-by-step."""
        self.decoder._set_initial_state(self.received_llr)
        for i in range(self.N):
            self._decoding_step(i)
