from unittest import TestCase

import numpy as np

from python_polar_coding.polar_codes.base import INFINITY
from python_polar_coding.polar_codes.base.functions import NodeTypes
from python_polar_coding.polar_codes.rc_scan import RCSCANNode


class TestRCSCANNode(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.llr = np.array([-2.7273, 8.7327, -0.1087, 1.6463, ])

    def test_zero_node(self):
        node = RCSCANNode(mask=np.zeros(4))
        self.assertTrue(node.is_zero)

        node.llr = self.llr
        node.initialize_leaf_beta()
        np.testing.assert_equal(
            node.beta,
            np.ones(4) * INFINITY,
        )

    def test_one_node(self):
        node = RCSCANNode(mask=np.ones(4))
        self.assertTrue(node.is_one)

        node.llr = self.llr
        node.initialize_leaf_beta()
        np.testing.assert_equal(
            node.beta,
            np.zeros(4)
        )

    def test_with_multiple_nodes(self):
        node = RCSCANNode(mask=np.array([
            0,
            1,
            0, 0,
            0, 0,
            1, 1,
            0, 0, 0, 0,
            1, 1, 1, 1,
        ]))

        leaf_path_lengths = [5, 5, 4, 4, 4, 3, 3]
        leaf_masks = [
            np.array([0, ]), np.array([1, ]), np.array([0, 0, ]),
            np.array([0, 0, ]), np.array([1, 1, ]),
            np.array([0, 0, 0, 0, ]), np.array([1, 1, 1, 1, ]),
        ]
        leaf_types = [
            NodeTypes.ZERO, NodeTypes.ONE, NodeTypes.ZERO,
            NodeTypes.ZERO, NodeTypes.ONE,
            NodeTypes.ZERO, NodeTypes.ONE,
        ]

        for i, leaf in enumerate(node.leaves):
            self.assertEqual(len(leaf.path), leaf_path_lengths[i])
            np.testing.assert_equal(leaf.mask, leaf_masks[i])
            self.assertTrue(leaf.node_type, leaf_types[i])
