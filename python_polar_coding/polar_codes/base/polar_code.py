import abc
from operator import itemgetter
from typing import Union

import numpy as np

from . import encoder, pcc, utils
from .crc import CRC


class BasicPolarCode(metaclass=abc.ABCMeta):
    """Basic Polar code class.

    Includes code construction.
    Provides the basic workflow for encoding and decoding.

    Supports creation of a polar code from custom mask.

    """
    encoder_class = encoder.Encoder
    decoder_class = None

    CUSTOM = 'custom'

    BHATTACHARYYA = 'bhattacharyya'
    GAUSSIAN = 'gaussian'
    MONTE_CARLO = 'monte_carlo'

    pcc_methods = {
        BHATTACHARYYA: pcc.bhattacharyya_bounds,
    }

    def __init__(self, N: int, K: int,
                 design_snr: float = 0.0,
                 is_systematic: bool = True,
                 custom_mask: Union[str, None] = None,
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
        self.mask = self._polar_code_construction(custom_mask)

        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()

    def get_encoder(self):
        """Get Polar Encoder instance."""
        return self.encoder_class(mask=self.mask, n=self.n,
                                  is_systematic=self.is_systematic)

    @abc.abstractmethod
    def get_decoder(self):
        """Get Polar Decoder instance."""

    def encode(self, message: np.array) -> np.array:
        """Encode binary message."""
        return self.encoder.encode(message)

    def decode(self, received_message: np.array) -> np.array:
        """Decode received message presented as LLR values."""
        return self.decoder.decode(received_message)

    def _compute_channels_estimates(self, N: int, n: int, design_snr: float,
                                    pcc_method: str):
        """Compute bit channel estimates for the polar code."""
        if pcc_method == self.CUSTOM:
            return None

        pcc_method = self.pcc_methods[pcc_method]
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
        """Build polar code Mask based on channel estimates.

        0 means frozen bit, 1 means information position.

        Supports bit-reversal approach, described in Section III-D of
        https://arxiv.org/abs/1307.7154

        """
        # represent each bit as tuple of 3 parameters:
        # (order, channel estimate, frozen / information position)
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


class BasicPolarCodeWithCRC(BasicPolarCode):
    """Basic Polar code class with CRC support.

    Provides the support of CRC 16 and CRC 32.

    """
    encoder_class = encoder.EncoderWithCRC

    def __init__(self, N: int, K: int,
                 design_snr: float = 0.0,
                 is_systematic: bool = True,
                 crc_size: int = 32,
                 custom_mask: Union[str, None] = None,
                 pcc_method: str = BasicPolarCode.BHATTACHARYYA):

        assert crc_size in [16, 32], f'Unsupported CRC size ({crc_size})'
        assert K + crc_size < N, (f'Cannot create Polar code with N = {N},'
                                  f' K = {K} and CRC {crc_size}.\n'
                                  f'N must be bigger than (K + CRC size).')
        self.crc_codec = CRC(crc_size)

        super().__init__(N=N, K=K,
                         is_systematic=is_systematic,
                         design_snr=design_snr,
                         custom_mask=custom_mask,
                         pcc_method=pcc_method)

    def get_encoder(self):
        """Get Polar Encoder instance."""
        return self.encoder_class(mask=self.mask, n=self.n,
                                  is_systematic=self.is_systematic,
                                  crc_codec=self.crc_codec)

    @property
    def crc_size(self):
        return self.crc_codec.crc_size if self.crc_codec else 0

    def _polar_code_construction(self, custom_mask=None) -> np.array:
        """Construct polar mask.

        If a mask was given as a string of 1s and 0s, it converts it to array.

        """
        if custom_mask:
            return np.array([int(b) for b in custom_mask])

        info_length = self.K + self.crc_size
        return self._construct_polar_mask(info_length)
