import numpy as np

from ..rc_scan import RCSCANDecoder
from .node import GFastSCANNode


class GFastSCANDecoder(RCSCANDecoder):

    node_class = GFastSCANNode

    def __init__(
            self,
            n: int,
            mask: np.array,
            AF: int = 1,
            I: int = 1,
    ):
        self.AF = AF
        super().__init__(n=n, mask=mask, I=I)

    def _setup_decoding_tree(self):
        """Setup decoding tree."""
        return self.node_class(mask=self.mask, AF=self.AF)
