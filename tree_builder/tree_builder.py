import json

import numpy as np

from polar_codes import SCPolarCode
from polar_codes.decoders.fast_ssc_decoder import FastSSCNode


class FastSSCTreeBuilder(FastSSCNode):
    """Tree builder for Fast SSC decoding research."""

    def __init__(self, node_min_size, channel_metrics, *args, **kwargs):
        self._min_node_size = node_min_size
        self.metrics = channel_metrics
        super().__init__(*args, **kwargs)

    @property
    def M(self):
        return self._min_node_size

    def to_dict(self):
        return {
            'type': self._node_type,
            'mask': ' '.join([str(i) for i in self._mask]),
            'metrics': np.array_str(self.metrics, precision=3),
            'pos': self.name,
        }

    def _get_node_type(self):
        """Get the type of Fast SSC Node.

        * Zero node - [0, 0, 0, 0, 0, 0, 0, 0];
        * One node - [1, 1, 1, 1, 1, 1, 1, 1];
        * Single parity check node - [0, 1, 1, 1, 1, 1, 1, 1];
        * Repetition node - [0, 0, 0, 0, 0, 0, 0, 1].

        Or other type.

        """
        if np.all(self._mask == 0):
            return FastSSCNode.ZERO_NODE
        if np.all(self._mask == 1):
            return FastSSCNode.ONE_NODE
        if self._mask[0] == 0 and np.sum(self._mask) == self.N - 1:
            return FastSSCNode.SINGLE_PARITY_CHECK
        if self._mask[-1] == 1 and np.sum(self._mask) == 1:
            return FastSSCNode.REPETITION
        return FastSSCNode.OTHER

    def _build_decoding_tree(self):
        """Build Fast SSC decoding tree."""
        if self.is_simplified_node:
            return

        if self.N == self.M:
            return

        left_mask, right_mask = np.split(self._mask, 2)
        left_metrics, right_metrics = np.split(self.metrics, 2)
        self.__class__(mask=left_mask, name=self.LEFT, parent=self,
                       node_min_size=self.M, channel_metrics=left_metrics)
        self.__class__(mask=right_mask, name=self.RIGHT, parent=self,
                       node_min_size=self.M, channel_metrics=right_metrics)


lengths = [int(pow(2, i)) for i in [10, 11, 12, 13, 14, 15, 16, ]]
code_speeds = [1/8, 1/4, 3/8, 1/2, 5/8, 3/4, 7/8, ]
design = [i / 2 for i in range(11)]
node_sizes = [8, 16, 32, 64, ]

parameters = [
    {
        'codeword_length': l,
        'info_length': int(l * c),
        'design_snr': d,
    }
    for l in lengths
    for c in code_speeds
    for d in design
]

trees = list()
for p in parameters:

    pc = SCPolarCode(**p)
    mask = pc.polar_mask
    estimates = pc.channel_estimates

    for n in node_sizes:
        tree_params = p.copy()
        tree_params['node_min_size'] = n

        tree = FastSSCTreeBuilder(
            mask=mask,
            channel_metrics=estimates,
            node_min_size=n
        )

        tree_params['leaves'] = [l.to_dict() for l in tree.leaves]

        trees.append(tree_params)
        print(p)

with open('polar_trees.json', 'w') as f:
    json.dump(trees, f)
