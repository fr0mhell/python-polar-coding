from typing import List, Tuple
import random
import numpy as np

from polar_codes import FastSSCPolarCode
from polar_codes.decoders.fast_ssc_decoder import FastSSCNode


class FastSSCTreeBuilder:
    """Tree builder for Fast SSC decoding research."""

    def __init__(self, codeword_length: int, info_length: int,
                 design_snr: float=0.0, is_crc_aided: bool=False,
                 dumped_mask=None, code_min_size: int=8):
        self.code = FastSSCPolarCode(
            codeword_length=codeword_length,
            info_length=info_length,
            design_snr=design_snr,
            is_crc_aided=is_crc_aided,
            dumped_mask=dumped_mask,
            code_min_size=code_min_size,
        )
        self._fast_ssc_sample_masks: dict = self._get_fast_ssc_minimum_codes()

    def __call__(self, *args, **kwargs):
        """Main method that performs tree rebuilding."""
        sub_codes_to_rebuild = self.get_sub_codes_to_rebuild()

        # initial data
        number_of_sub_codes = len(sub_codes_to_rebuild)
        K_to_cover = np.sum([np.sum(s['mask']) for s in sub_codes_to_rebuild])
        mask_samples = self.samples.copy()
        positions = [s['position'] for s in sub_codes_to_rebuild]
        overall_metric = np.array([
            s['metrics'] for s in sub_codes_to_rebuild
        ]).flatten()

        # Generate initial population
        initial_population_size = number_of_sub_codes * len(mask_samples) * 10

        combinations = []
        while True:
            combination = random.choices(
                list(mask_samples.keys()), k=number_of_sub_codes)

            if sum(combination) == K_to_cover:
                combinations.append(combination)
            if len(combinations) == initial_population_size:
                break

        for comb in combinations:
            result_mask = np.array([mask_samples[i] for i in comb]).flatten()
            comb_metric = np.sum(result_mask * overall_metric)

        # Mutate:
        # 1. Split into 2 parts. Randomly select sub-code1 and sub-code2. Swap.
        # 2. Copy combination. Shuffle copy.
        # 3. Select random value. Make left/right cyclic shift. See np.roll()

    @property
    def Ns(self):
        """Minimal size of component polar code."""
        return self.code.decoder.Ns

    @property
    def samples(self):
        """Samples of minimal Fast SSC compatible polar sub-codes."""
        return self._fast_ssc_sample_masks

    def get_sub_codes_to_rebuild(self):
        """Get sub codes for further rebuilding."""
        sub_codes = self._get_sub_codes()
        fast_ssc, other = self._group_sub_codes_by_type(sub_codes)

        while True:
            if self._check_other_sub_codes_can_be_rebuilt(other):
                return other
            fast_ssc, other = self._update_other_sub_codes(fast_ssc, other)

    def _get_fast_ssc_minimum_codes(self) -> dict:
        """Get examples of minimal component codes."""
        repetition = np.zeros(self.Ns, dtype=np.int8)
        repetition[-1] = 1

        spc = np.ones(self.Ns, dtype=np.int8)
        spc[0] = 0

        result = {
            0: np.zeros(self.Ns, dtype=np.int8),
            1: repetition,
            self.Ns - 1: spc,
            self.Ns: np.ones(self.Ns, dtype=np.int8)
        }
        return result

    def _get_sub_codes(self) -> List:
        """Get sub-codes."""
        sub_codes = list()
        itr: int = 0
        for i, l in enumerate(self.code.decoder._decoding_tree.leaves):
            leaf = l.to_dict()
            leaf.update({
                'position': i,
                'metrics': self.code.channel_estimates[itr: itr + l.N]
            })
            sub_codes.append(leaf)
            itr += l.N

        return sub_codes

    def _group_sub_codes_by_type(self, sub_codes: list) -> Tuple[List, List]:
        """Split sub-codes into two groups.

        1. Fast SSC compatible sub-codes;
        2. Other sub-codes.

        """
        other = [s for s in sub_codes if s['type'] == FastSSCNode.OTHER]
        fast_ssc = [s for s in sub_codes if s['type'] != FastSSCNode.OTHER]
        fast_ssc = sorted(
            fast_ssc, key=lambda code: len(code['mask']), reverse=True
        )
        fast_ssc = sorted(
            fast_ssc,
            key=lambda code: np.sum(np.array(code['mask']) * np.array(code['metrics']))  # noqa
        )

        return fast_ssc, other

    def _check_other_sub_codes_can_be_rebuilt(self, other_sub_codes: list) -> bool:
        """"""
        N_sub_code = len(other_sub_codes[0]['mask'])
        code_bits_to_cover = len(other_sub_codes) * N_sub_code
        info_bits_to_cover = np.sum([np.sum(o['mask']) for o in other_sub_codes])  # noqa
        dividers = sorted(self.samples.keys(), reverse=True)

        # exclude 0
        for i, d in enumerate(dividers[:-1]):
            covered_sub_codes = info_bits_to_cover // d
            code_bits_to_cover -= covered_sub_codes * N_sub_code
            info_bits_to_cover -= covered_sub_codes * d

            if code_bits_to_cover < 0:
                return False
            if info_bits_to_cover == 0:
                return True
        return False

    def _update_other_sub_codes(self, fast_ssc_sub_codes: list,
                                other_sub_codes: list) -> Tuple[List, List]:
        """If non Fast SSC sub-codes cannot be rebuilt to Fast SSC ones,
        extend them with Fast SSC sub-codes with highest metrics.

        """
        while True:
            if self._check_other_sub_codes_can_be_rebuilt(other_sub_codes):
                break
            other_sub_codes.append(fast_ssc_sub_codes.pop(-1))

        return fast_ssc_sub_codes, other_sub_codes
