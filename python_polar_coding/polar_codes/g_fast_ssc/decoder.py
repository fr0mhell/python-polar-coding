import numpy as np

from python_polar_coding.polar_codes.fast_ssc import FastSSCDecoder

from .node import GeneralizedFastSSCNode


class GeneralizedFastSSCDecoder(FastSSCDecoder):
    node_class = GeneralizedFastSSCNode

    def __init__(
            self,
            n: int,
            mask: np.array,
            is_systematic: bool = True,
            code_min_size: int = 0,
            AF: int = 1,
    ):
        self.AF = AF
        super().__init__(
            n=n,
            mask=mask,
            is_systematic=is_systematic,
            code_min_size=code_min_size,
        )

    def setup_decoding_tree(self, N_min, **kwargs):
        """Setup decoding tree."""
        return self.node_class(mask=self.mask, N_min=N_min, AF=self.AF)
