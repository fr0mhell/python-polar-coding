import numpy as np

from python_polar_coding.polar_codes.fast_ssc import FastSSCDecoder

from .node import GFastSSCNode


class GFastSSCDecoder(FastSSCDecoder):
    node_class = GFastSSCNode

    def __init__(self, n: int, mask: np.array, AF: int = 0):
        self.AF = AF
        super().__init__(n=n, mask=mask)

    def _setup_decoding_tree(self):
        """Setup decoding tree."""
        return self.node_class(mask=self.mask, AF=self.AF)
