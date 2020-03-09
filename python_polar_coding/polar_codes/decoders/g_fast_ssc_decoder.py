import numpy as np

from ..base import functions
from .fast_ssc_decoder import FastSSCDecoder, FastSSCNode


def splits(start, end):
    while start <= end:
        yield start
        start *= 2


class GeneralizedFastSSCNode(FastSSCNode):
    """Decoder for Generalized Fast SSC code.

    Based on: https://arxiv.org/pdf/1804.09508.pdf

    """
    G_REPETITION = 'G-REPETITION'
    RG_PARITY = 'RG_PARITY'

    CHUNK_SIZE = 4

    def __init__(self, AF=1, *args, **kwargs):
        self.AF = AF
        self.last_chunk_type = None
        self.mask_chunk = None
        super().__init__(*args, **kwargs)

    def get_node_type(self):
        ntype = super().get_node_type()
        if ntype != self.OTHER:
            return ntype
        if self._check_is_g_repetition():
            return self.G_REPETITION
        if self._check_is_rg_parity():
            return self.RG_PARITY
        return self.OTHER

    def _check_is_g_repetition(self):
        """Check the node is Generalized Repetition node.

        Based on: https://arxiv.org/pdf/1804.09508.pdf, Section III, A.

        """
        # 1. Split mask into T chunks, T in range [4, 8, ..., N/2]
        for t in list(splits(self.__class__.CHUNK_SIZE, self.N//2)):
            chunks = np.split(self._mask, self.N//t)

            last_ok = (self._check_is_spc(chunks[-1]) or
                       self._check_is_one(chunks[-1]))
            if not last_ok:
                continue

            others_ok = all(self._check_is_zero(c) for c in chunks[:-1])
            if not others_ok:
                continue

            self.last_chunk_type = 1 if self._check_is_one(chunks[-1]) else 0
            self.mask_chunk = chunks[-1].size
            return True

        return False

    def _check_is_rg_parity(self):
        """Check the node is Relaxed Generalized Parity Check node.

        Based on: https://arxiv.org/pdf/1804.09508.pdf, Section III, B.

        """
        # 1. Split mask into T chunks, T in range [4, 8, ..., N/2]
        for t in list(splits(self.__class__.CHUNK_SIZE, self.N//2)):
            chunks = np.split(self._mask, self.N//t)

            if not self._check_is_zero(chunks[0]):
                continue

            ones = 0
            spcs = 0
            for c in chunks[1:]:
                if self._check_is_one(c):
                    ones += 1
                if self._check_is_spc(c):
                    spcs += 1

            others_ok = (ones + spcs + 1) * t == self.N and spcs <= self.AF
            if not others_ok:
                continue

            self.mask_chunk = chunks[0].size
            return True

        return False

    def compute_leaf_beta(self):
        super().compute_leaf_beta()
        klass = self.__class__

        if self._node_type == klass.G_REPETITION:
            self._beta = functions.compute_g_repetition(
                llr=self.alpha,
                mask_chunk=self.mask_chunk,
                last_chunk_type=self.last_chunk_type,
                N=self.N,
            )
        if self._node_type == klass.RG_PARITY:
            self._beta = functions.compute_rg_parity(
                llr=self.alpha,
                mask_chunk=self.mask_chunk,
                N=self.N,
            )


class GeneralizedFastSSCDecoder(FastSSCDecoder):
    node_class = GeneralizedFastSSCNode

    def __init__(self, n: int,
                 mask: np.array,
                 is_systematic: bool = True,
                 code_min_size: int = 0,
                 AF: int = 1):
        super().__init__(n=n, mask=mask, is_systematic=is_systematic)
        self._decoding_tree = self.node_class(mask=self.mask,
                                              code_min_size=code_min_size,
                                              AF=AF)
        self._position = 0
