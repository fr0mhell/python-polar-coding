from unittest import TestCase

import numpy as np

from polar_codes.successive_cancellation.sc_decoder import SCDecoder


class TestSCDecoder(TestCase):

    def setUp(self):
        self.message = np.array([
            1.01006308,
            -0.63763626,
            0.70333425,
            -3.28018935,
            1.19738423,
            -0.2126217,
            -0.48246313,
            -1.44867097
        ])
        self.decoder = SCDecoder(self.message)

    def test_initial_params(self):
        self.assertEqual(self.decoder.n, 3)

        start = self.decoder.N
        for stage in self.decoder.intermediate_llr:
            start = start // 2
            self.assertEqual(stage.size, start)

    def test_compute_intermediate_llr_for_zero_level(self):
        """Test `compute_intermediate_llr` method for zero level.

        Zero level means decoding of the first symbol.

        """
        expected = [
            np.array([-0.63763626, -0.70333425, -0.2126217, 0.48246313]),
            np.array([0.63763626, -0.2126217]),
            np.array([-0.2126217]),
        ]

        self.decoder.set_decoder_state()

        self.decoder.compute_intermediate_llr()

        for i in range(self.decoder.n):
            np.testing.assert_array_almost_equal(
                self.decoder.intermediate_llr[i],
                expected[i]
            )
