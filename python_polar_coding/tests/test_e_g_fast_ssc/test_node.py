from unittest import TestCase

import numpy as np

from python_polar_coding.polar_codes.e_g_fast_ssc import EGFastSSCNode


class TestEGFastSSC(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.check_node = EGFastSSCNode(mask=np.zeros(4))
        cls.alpha8 = np.array([

        ])

    # ZERO-ANY node

    def test_check_is_zero_any_right_g_rep(self):
        """"""
        mask = np.array([
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 1,
        ])
        self.assertTrue(self.check_node._check_is_zero_any(mask))

    def test_check_is_zero_any_right_rg_par(self):
        """"""
        mask = np.array([
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 1, 1, 1, 1, 1,
        ])
        self.assertTrue(self.check_node._check_is_zero_any(mask))

    def test_compute_zero_any_right_rg_parity(self):
        """"""
        mask = np.array([0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 1, 1, 1, 1, 1, 1])
        alpha = np.array([
            -2.7273, -8.7327,  0.1087, -1.6463,
             2.7273, -8.7327, -0.1087,  1.6463,
            -2.7273, -8.7327, -0.1087,  1.6463,
             2.7273,  8.7326,  1.1087, -1.6463,
        ])
        expected = np.array([1, 1, 1, 0, 0, 1, 0, 0,
                             1, 1, 1, 0, 0, 1, 0, 0, ])

        node = EGFastSSCNode(mask=mask)
        node.alpha = alpha
        node.compute_leaf_beta()

        np.testing.assert_equal(node.beta, expected)

    # REP-ANY node

    def test_check_is_rep_any_right_one(self):
        """"""
        mask = np.array([0, 0, 0, 1, 1, 1, 1, 1])
        self.assertTrue(self.check_node._check_is_rep_any(mask))

    def test_check_is_rep_any_right_spc(self):
        """"""
        mask = np.array([0, 0, 0, 1, 0, 1, 1, 1])
        self.assertTrue(self.check_node._check_is_rep_any(mask))

    def test_check_is_rep_any_right_g_rep(self):
        """"""
        mask = np.array([
            0, 0, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 0, 1, 1,
        ])
        self.assertTrue(self.check_node._check_is_rep_any(mask))

    def test_check_is_rep_any_right_rg_par(self):
        """"""
        mask = np.array([
            0, 0, 0, 0, 0, 0, 0, 1,
            0, 0, 1, 1, 1, 1, 1, 1,
        ])
        self.assertTrue(self.check_node._check_is_rep_any(mask))
