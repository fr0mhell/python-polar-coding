import abc
from typing import Union
from operator import itemgetter

import numpy as np

from python_polar_coding.polar_codes import pcc, utils

from . import encoder


class BasePolarCodec(metaclass=abc.ABCMeta):
    """Basic codec for Polar code.

    Includes code construction.
    Defines the basic workflow for encoding and decoding.

    Supports creation of a polar code from custom mask.

    """
    encoder_class = encoder.Encoder
    decoder_class = None

    BHATTACHARYYA = 'bhattacharyya'
    GAUSSIAN = 'gaussian'
    MONTE_CARLO = 'monte carlo'

    PCC_METHODS = {
        BHATTACHARYYA: pcc.bhattacharyya_bounds,
    }

    def __init__(self, N: int, K: int,
                 design_snr: float = 0.0,
                 is_systematic: bool = True,
                 mask: Union[str, None] = None,
                 pcc_method: str = BHATTACHARYYA):

        assert K < N, (f'Cannot create Polar code with N = {N}, K = {K}.'
                       f'\nN must be bigger than K.')

        self.N = N
        self.K = K
        self.n = int(np.log2(N))
        self.design_snr = design_snr
        self.is_systematic = is_systematic

        self.pcc_method = pcc_method
        self.channel_estimates = self._compute_channels_estimates(
            N=self.N, n=self.n, design_snr=design_snr, pcc_method=pcc_method)
        self.mask = self._polar_code_construction(mask)

        self.encoder = self.init_encoder()
        self.decoder = self.init_decoder()

    def __str__(self):
        return (f'({self.N}, {self.K}) Polar code.\n'
                f'Design SNR: {self.design_snr} dB\n'
                f'Systematic: {str(self.is_systematic)}\n')

    def to_dict(self):
        """Get code parameters as a dict."""
        return {
            'type': self.__class__.__name__,
            'N': self.N,
            'K': self.K,
            'is_systematic': self.is_systematic,
            'design_snr': self.design_snr,
            'pcc_method': self.pcc_method,
            'mask': ''.join(str(m) for m in self.mask),
        }

    def init_encoder(self):
        """Get Polar Encoder instance."""
        return self.encoder_class(mask=self.mask, n=self.n,
                                  is_systematic=self.is_systematic)

    @abc.abstractmethod
    def init_decoder(self):
        """Get Polar Decoder instance."""

    def encode(self, message: np.array) -> np.array:
        """Encode binary message."""
        return self.encoder.encode(message)

    def decode(self, received_message: np.array) -> np.array:
        """Decode received message presented as LLR values."""
        return self.decoder.decode(received_message)

    def _compute_channels_estimates(self, N: int, n: int, design_snr: float,
                                    pcc_method: str):
        """Compute bit channels estimates for the polar code."""
        if pcc_method not in self.PCC_METHODS.keys():
            return None

        pcc_method = self.PCC_METHODS[pcc_method]
        channel_estimates = pcc_method(N, design_snr)

        # bit-reversal approach https://arxiv.org/abs/1307.7154 (Section III-D)
        return np.array([
            channel_estimates[utils.reverse_bits(i, n)] for i in range(N)
        ])

    def _polar_code_construction(self, custom_mask=None) -> np.array:
        """Construct polar mask.

        If a mask was given as a string of 1s and 0s, it converts it to array.

        """
        if custom_mask:
            return np.array([int(b) for b in custom_mask])
        return self._construct_polar_mask(self.K)

    def _construct_polar_mask(self, K):
        """Build polar code Mask based on channels estimates.

        0 means frozen bit, 1 means information position.

        Supports bit-reversal approach, described in Section III-D of
        https://arxiv.org/abs/1307.7154

        """
        # represent each bit as tuple of 3 parameters:
        # (order, channels estimate, frozen / information position)
        mask = [[i, b, 0] for i, b in enumerate(self.channel_estimates)]

        # sort channels due to estimates
        mask = sorted(mask, key=itemgetter(1))
        # set information position for first `info_length` channels
        for m in mask[:K]:
            m[2] = 1
        # sort channels due to order
        mask = sorted(mask, key=itemgetter(0))
        # return mask, contains 0s or 1s due to frozen/info position
        return np.array([m[2] for m in mask])
