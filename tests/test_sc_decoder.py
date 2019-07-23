from unittest import TestCase

import numpy as np

from polar_codes.successive_cancellation.sc_decoder import SCDecoder


class TestSCDecoder(TestCase):

    @classmethod
    def setUp(cls):
        cls.message = np.array([
            1.01006308,
            -0.63763626,
            0.70333425,
            -3.28018935,
            1.19738423,
            -0.2126217,
            -0.48246313,
            -1.44867097
        ])
        cls.mask = np.array([0, 1, 0, 1, 0, 1, 1, 1, ], dtype=np.int8)
        cls.decoded = np.array([0, 0, 0, 1, 0, 0, 0, 1], dtype=np.int8)
        cls.decoder = SCDecoder(cls.message, cls.mask)

    def test_initial_params(self):
        self.assertEqual(self.decoder.n, 3)

        start = self.decoder.N
        for stage in self.decoder.intermediate_llr:
            start = start // 2
            self.assertEqual(stage.size, start)

    def test_decoding_process(self):
        """Test SC decoding process step-by-step."""
        # Step 0
        self.decoder.set_decoder_state(0)

        expected_llr = [
            np.array([-0.63763626, -0.70333425, -0.2126217, 0.48246313]),
            np.array([0.63763626, -0.2126217]),
            np.array([-0.2126217]),
        ]
        self.decoder.compute_intermediate_llr()
        for i in range(self.decoder.n):
            np.testing.assert_array_almost_equal(
                self.decoder.intermediate_llr[i],
                expected_llr[i]
            )

        self.decoder.make_decision()
        self.assertEqual(self.decoder.decoded[self.decoder.current_position], 0)

        expected_bits = [
            np.array([-1, -1, -1, -1, -1, -1, -1, -1, ]),
            np.array([-1, -1, -1, -1, -1, -1, -1, -1, ]),
            np.array([0, -1, -1, -1, -1, -1, -1, -1, ]),
        ]
        self.decoder.compute_intermediate_bits()
        for i in range(self.decoder.n):
            np.testing.assert_equal(
                self.decoder.intermediate_bits[i],
                expected_bits[i]
            )

        self.decoder.set_next_decoding_position()

        # Step 1
        self.decoder.set_decoder_state(1)

        expected_llr = [
            np.array([-0.63763626, -0.70333425, -0.2126217, 0.48246313]),
            np.array([0.63763626, -0.2126217]),
            np.array([0.42501456]),
        ]
        self.decoder.compute_intermediate_llr()
        for i in range(self.decoder.n):
            np.testing.assert_array_almost_equal(
                self.decoder.intermediate_llr[i],
                expected_llr[i]
            )

        self.decoder.make_decision()
        self.assertEqual(self.decoder.decoded[self.decoder.current_position], 0)

        expected_bits = [
            np.array([-1, -1, -1, -1, -1, -1, -1, -1, ]),
            np.array([0, 0, -1, -1, -1, -1, -1, -1, ]),
            np.array([0, 0, -1, -1, -1, -1, -1, -1, ]),
        ]
        self.decoder.compute_intermediate_bits()
        for i in range(self.decoder.n):
            np.testing.assert_equal(
                self.decoder.intermediate_bits[i],
                expected_bits[i]
            )

        self.decoder.set_next_decoding_position()

        # Step 2
        self.decoder.set_decoder_state(2)

        expected_llr = [
            np.array([-0.63763626, -0.70333425, -0.2126217, 0.48246313]),
            np.array([-1.34097051, 0.26984143]),
            np.array([-0.26984143]),
        ]
        self.decoder.compute_intermediate_llr()
        for i in range(self.decoder.n):
            np.testing.assert_array_almost_equal(
                self.decoder.intermediate_llr[i],
                expected_llr[i]
            )

        self.decoder.make_decision()
        self.assertEqual(self.decoder.decoded[self.decoder.current_position], 0)

        expected_bits = [
            np.array([-1, -1, -1, -1, -1, -1, -1, -1, ]),
            np.array([0, 0, -1, -1, -1, -1, -1, -1, ]),
            np.array([0, 0, 0, -1, -1, -1, -1, -1, ]),
        ]
        self.decoder.compute_intermediate_bits()
        for i in range(self.decoder.n):
            np.testing.assert_equal(
                self.decoder.intermediate_bits[i],
                expected_bits[i]
            )

        self.decoder.set_next_decoding_position()

        # Step 3
        self.decoder.set_decoder_state(3)

        expected_llr = [
            np.array([-0.63763626, -0.70333425, -0.2126217, 0.48246313]),
            np.array([-1.34097051, 0.26984143]),
            np.array([-1.07112908]),
        ]
        self.decoder.compute_intermediate_llr()
        for i in range(self.decoder.n):
            np.testing.assert_array_almost_equal(
                self.decoder.intermediate_llr[i],
                expected_llr[i]
            )

        self.decoder.make_decision()
        self.assertEqual(self.decoder.decoded[self.decoder.current_position], 1)

        expected_bits = [
            np.array([1, 1, 1, 1, -1, -1, -1, -1, ]),
            np.array([0, 0, 1, 1, -1, -1, -1, -1, ]),
            np.array([0, 0, 0, 1, -1, -1, -1, -1, ]),
        ]
        self.decoder.compute_intermediate_bits()
        for i in range(self.decoder.n):
            np.testing.assert_equal(
                self.decoder.intermediate_bits[i],
                expected_bits[i]
            )

        self.decoder.set_next_decoding_position()

        # Step 4
        self.decoder.set_decoder_state(4)

        expected_llr = [
            np.array([-1.64769934, -3.9835236, -1.41000593, -0.96620784]),
            np.array([1.64769934, 0.96620784]),
            np.array([0.96620784]),
        ]
        self.decoder.compute_intermediate_llr()
        for i in range(self.decoder.n):
            np.testing.assert_array_almost_equal(
                self.decoder.intermediate_llr[i],
                expected_llr[i]
            )

        self.decoder.make_decision()
        self.assertEqual(self.decoder.decoded[self.decoder.current_position], 0)

        expected_bits = [
            np.array([1, 1, 1, 1, -1, -1, -1, -1, ]),
            np.array([0, 0, 1, 1, -1, -1, -1, -1, ]),
            np.array([0, 0, 0, 1, 0, -1, -1, -1, ]),
        ]
        self.decoder.compute_intermediate_bits()
        for i in range(self.decoder.n):
            np.testing.assert_equal(
                self.decoder.intermediate_bits[i],
                expected_bits[i]
            )

        self.decoder.set_next_decoding_position()

        # Step 5
        self.decoder.set_decoder_state(5)

        expected_llr = [
            np.array([-1.64769934, -3.9835236, -1.41000593, -0.96620784]),
            np.array([1.64769934, 0.96620784]),
            np.array([2.61390718]),
        ]
        self.decoder.compute_intermediate_llr()
        for i in range(self.decoder.n):
            np.testing.assert_array_almost_equal(
                self.decoder.intermediate_llr[i],
                expected_llr[i]
            )

        self.decoder.make_decision()
        self.assertEqual(self.decoder.decoded[self.decoder.current_position], 0)

        expected_bits = [
            np.array([1, 1, 1, 1, -1, -1, -1, -1, ]),
            np.array([0, 0, 1, 1, 0, 0, -1, -1, ]),
            np.array([0, 0, 0, 1, 0, 0, -1, -1, ]),
        ]
        self.decoder.compute_intermediate_bits()
        for i in range(self.decoder.n):
            np.testing.assert_equal(
                self.decoder.intermediate_bits[i],
                expected_bits[i]
            )

        self.decoder.set_next_decoding_position()

        # Step 6
        self.decoder.set_decoder_state(6)

        expected_llr = [
            np.array([-1.64769934, -3.9835236, -1.41000593, -0.96620784]),
            np.array([-5.63122294, -2.37621377]),
            np.array([2.37621377]),
        ]
        self.decoder.compute_intermediate_llr()
        for i in range(self.decoder.n):
            np.testing.assert_array_almost_equal(
                self.decoder.intermediate_llr[i],
                expected_llr[i]
            )

        self.decoder.make_decision()
        self.assertEqual(self.decoder.decoded[self.decoder.current_position], 0)

        expected_bits = [
            np.array([1, 1, 1, 1, -1, -1, -1, -1, ]),
            np.array([0, 0, 1, 1, 0, 0, -1, -1, ]),
            np.array([0, 0, 0, 1, 0, 0, 0, -1, ]),
        ]
        self.decoder.compute_intermediate_bits()
        for i in range(self.decoder.n):
            np.testing.assert_equal(
                self.decoder.intermediate_bits[i],
                expected_bits[i]
            )

        self.decoder.set_next_decoding_position()

        # Step 7
        self.decoder.set_decoder_state(7)

        expected_llr = [
            np.array([-1.64769934, -3.9835236, -1.41000593, -0.96620784]),
            np.array([-5.63122294, -2.37621377]),
            np.array([-8.00743671]),
        ]
        self.decoder.compute_intermediate_llr()
        for i in range(self.decoder.n):
            np.testing.assert_array_almost_equal(
                self.decoder.intermediate_llr[i],
                expected_llr[i]
            )

        self.decoder.make_decision()
        self.assertEqual(self.decoder.decoded[self.decoder.current_position], 1)

        np.testing.assert_array_equal(self.decoder.decoded, self.decoded)

    def test_decoding_step_by_step(self):
        """Test `decoder_step` method."""
        for i in range(self.message.size):
            self.decoder.decoder_step(i)

        np.testing.assert_array_equal(self.decoder.decoded, self.decoded)
