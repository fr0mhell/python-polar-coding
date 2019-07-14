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
