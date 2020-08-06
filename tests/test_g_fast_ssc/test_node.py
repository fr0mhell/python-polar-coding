from unittest import TestCase

import numpy as np

from python_polar_coding.polar_codes.base import NodeTypes
from python_polar_coding.polar_codes.g_fast_ssc import GFastSSCNode


class GeneralizedFastSSCNodeTest(TestCase):

    def setUp(self):
        self.llr = np.array([
            -2.7273, -8.7327,  0.1087, -1.6463,
             2.7273, -8.7327, -0.1087,  1.6463,
            -2.7273, -8.7327, -0.1087,  1.6463,
             2.7273,  8.7326,  1.1087, -1.6463,
        ])
        self.g_rep_one = np.array([
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
        ])
        self.g_rep_one_long = np.array([
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        ])
        self.g_rep_spc = np.array([
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
        ])
        self.g_rep_spc_long = np.array([
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
        ])

        self.rg_parity_ones = np.array([
            0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ])
        self.rg_parity_ones_long = np.array([
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ])

        self.rg_parity_ones_spc = np.array([
            0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
        ])
        self.rg_parity_ones_spc_long = np.array([
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        ])

    def test_g_repetition_one(self):
        node = GFastSSCNode(mask=self.g_rep_one)
        self.assertEqual(node.node_type, NodeTypes.G_REPETITION)

        expected_result = np.array([0, 1, 0, 0,
                                    0, 1, 0, 0,
                                    0, 1, 0, 0,
                                    0, 1, 0, 0, ])

        node.alpha = self.llr
        node()
        np.testing.assert_array_equal(node.beta, expected_result)

    def test_g_repetition_one_long(self):
        node = GFastSSCNode(mask=self.g_rep_one_long)
        self.assertEqual(node.node_type, NodeTypes.G_REPETITION)

    def test_g_repetition_spc(self):
        node = GFastSSCNode(mask=self.g_rep_spc)
        self.assertEqual(node.node_type, NodeTypes.G_REPETITION)

        expected_result = np.array([1, 1, 0, 0,
                                    1, 1, 0, 0,
                                    1, 1, 0, 0,
                                    1, 1, 0, 0, ])

        node.alpha = self.llr
        node()
        np.testing.assert_array_equal(node.beta, expected_result)

    def test_g_repetition_spc_long(self):
        node = GFastSSCNode(mask=self.g_rep_spc_long)
        self.assertEqual(node.node_type, NodeTypes.G_REPETITION)

    def test_rg_parity_ones(self):
        node = GFastSSCNode(mask=self.rg_parity_ones)
        self.assertEqual(node.node_type, NodeTypes.RG_PARITY)

        expected_result = np.array([1, 1, 0, 1,
                                    0, 1, 1, 0,
                                    1, 1, 1, 0,
                                    0, 1, 0, 1, ])

        node.alpha = self.llr
        node()
        np.testing.assert_array_equal(node.beta, expected_result)

    def test_rg_parity_ones_long(self):
        node = GFastSSCNode(mask=self.rg_parity_ones_long)
        self.assertEqual(node.node_type, NodeTypes.RG_PARITY)

    def test_rg_parity_spc_ones_spc_AF_1(self):
        node = GFastSSCNode(mask=self.rg_parity_ones_spc, AF=1)
        self.assertEqual(node.node_type, NodeTypes.RG_PARITY)

        expected_result = np.array([1, 1, 0, 1,
                                    0, 1, 1, 0,
                                    1, 1, 1, 0,
                                    0, 1, 0, 1, ])

        node.alpha = self.llr
        node()
        np.testing.assert_array_equal(node.beta, expected_result)

    def test_rg_parity_spc_ones_long_AF_1(self):
        node = GFastSSCNode(mask=self.rg_parity_ones_spc_long, AF=1)
        self.assertEqual(node.node_type, NodeTypes.RG_PARITY)

    def test_rg_parity_spc_ones_spc_AF_2(self):
        mask = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, ])
        node = GFastSSCNode(mask=mask)
        self.assertNotEqual(node.node_type, NodeTypes.RG_PARITY)

        node = GFastSSCNode(mask=mask, AF=2)
        self.assertEqual(node.node_type, NodeTypes.RG_PARITY)

    def test_rg_parity_spc_ones_spc_AF_3(self):
        mask = np.array([0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, ])
        node = GFastSSCNode(mask=mask, AF=2)
        self.assertNotEqual(node.node_type, NodeTypes.RG_PARITY)

        node = GFastSSCNode(mask=mask, AF=3)
        self.assertEqual(node.node_type, NodeTypes.RG_PARITY)
