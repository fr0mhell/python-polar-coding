from typing import Union

from .decoders.g_fast_ssc_decoder import GeneralizedFastSSCDecoder
from .fast_ssc import FastSSCPolarCode


class GeneralizedFastSSCPolarCode(FastSSCPolarCode):
    """Generalized Fast SSC code.

    Based on: https://arxiv.org/pdf/1804.09508.pdf

    """
    decoder_class = GeneralizedFastSSCDecoder

    def __init__(self, N: int, K: int,
                 design_snr: float = 0.0,
                 is_systematic: bool = True,
                 mask: Union[str, None] = None,
                 pcc_method: str = FastSSCPolarCode.BHATTACHARYYA,
                 Ns: int = 1,
                 AF: int = 1):

        self.Ns = Ns
        self.AF = AF
        super().__init__(N=N, K=K,
                         is_systematic=is_systematic,
                         design_snr=design_snr,
                         mask=mask,
                         pcc_method=pcc_method)

    def get_decoder(self):
        return self.decoder_class(n=self.n, mask=self.mask,
                                  is_systematic=self.is_systematic,
                                  code_min_size=self.Ns,
                                  AF=self.AF)

    def to_dict(self):
        d = super().to_dict()
        d.update({'AF': self.AF})
        return d

    def show_tree(self):
        nodes = [
            (f'{leaf.name}, {leaf._node_type}, {leaf.N}: '
             f'{"".join([str(m) for m in leaf._mask])}')
            for leaf in self.tree.leaves
        ]
        return '\n'.join(nodes)
