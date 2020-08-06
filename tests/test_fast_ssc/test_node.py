from unittest import TestCase

import numpy as np

from python_polar_coding.polar_codes.base import NodeTypes
from python_polar_coding.polar_codes.fast_ssc.decoder import FastSSCNode


class FastSSCNodeTest(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.llr = np.array([-2.7273, 8.7327, -0.1087, 1.6463, ])

    def test_zero_node(self):
        node = FastSSCNode(np.zeros(4))

        node.alpha = self.llr
        node()
        np.testing.assert_equal(node.beta, np.zeros(4))

    def test_one_node(self):
        node = FastSSCNode(np.ones(4))

        node.alpha = self.llr
        node()
        np.testing.assert_equal(node.beta, np.array([1, 0, 1, 0]))

    def test_spc_node(self):
        node = FastSSCNode(np.array([0, 1, 1, 1]))

        node.alpha = self.llr
        node()
        np.testing.assert_equal(node.beta, np.array([1, 0, 1, 0]))

    def test_repetition_node(self):
        node = FastSSCNode(np.array([0, 0, 0, 1]))

        node.alpha = self.llr
        node()
        np.testing.assert_equal(node.beta, np.array([0, 0, 0, 0]))

    def test_with_multiple_nodes(self):
        node = FastSSCNode(np.array([
            1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1,
        ]))

        leaf_path_lengths = [4, 4, 3, 5, 5, 5, 5, 3]
        leaf_masks = [
            np.array([1, 1]),
            np.array([0, 1]),
            np.array([0, 0, 0, 1]),
            np.array([1]),
            np.array([0]),
            np.array([1]),
            np.array([0]),
            np.array([0, 1, 1, 1]),
        ]
        leaf_types = [
            NodeTypes.ONE,
            NodeTypes.REPETITION,
            NodeTypes.REPETITION,
            NodeTypes.ONE,
            NodeTypes.ZERO,
            NodeTypes.ONE,
            NodeTypes.ZERO,
            NodeTypes.SINGLE_PARITY_CHECK,
        ]

        for i, leaf in enumerate(node.leaves):
            self.assertEqual(len(leaf.path), leaf_path_lengths[i])
            np.testing.assert_equal(leaf.mask, leaf_masks[i])
            self.assertEqual(leaf.node_type, leaf_types[i])
