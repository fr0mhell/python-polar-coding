from operator import itemgetter

import numpy as np
from typing import Any
from ..utils import (calculate_crc_16, check_crc_16, int_to_bin_list,
                     reverse_bits)
from .functions import compute_encoding_step
from .polar_code_construction import bhattacharyya_bounds


class BasicPolarCode:
    """Base Polar code with CRC support.

    Include code construction, encoding and CRC 16 support.

    """
    CRC_SIZE = 16

    CUSTOM = 'custom'
    BHATTACHARYYA = 'bhattacharyya'
    GAUSSIAN = 'gaussian'
    MONTE_CARLO = 'monte_carlo'

    pcc_methods = {
        BHATTACHARYYA: bhattacharyya_bounds,
    }

    def __init__(self, N: int, K: int,
                 design_snr: float = 0.0,
                 is_systematic: bool = True,
                 is_crc_aided: bool = False,
                 mask: Any[str, None] = None,
                 pcc_method: str = BHATTACHARYYA):

        assert K < N, (f'Cannot create Polar code with N = {N}, K = {K}.'
                       f'\nN must be bigger than K.')

        self._N = N
        self._K = K
        self._n = self._calculate_polar_steps(N)
        self.design_snr = design_snr
        self.is_systematic = is_systematic
        self.is_crc_aided = is_crc_aided

        self.pcc_method = pcc_method
        self.channel_estimates = self.compute_channels_estimates(self.pcc_method)
        self.polar_mask = self.polar_code_construction(mask)

        self.decoder = None

    @property
    def N(self):
        """Get codeword length using the common name `N`."""
        return self._N

    @property
    def K(self):
        """Get information word length using the common name `K`."""
        return self._K

    @property
    def n(self):
        """Get the number of polar steps using the common name `n`."""
        return self._n

    def __str__(self):
        return (
            f'({self.N}, {self.K}) Polar code.\n'
            f'Design SNR: {self.design_snr} dB\n'
            f'Systematic: {str(self.is_systematic)}\n'
            f'CRC aided: {str(self.is_crc_aided)}\n'
        )

    def _get_extra_params(self):
        return dict()

    def to_dict(self):
        """Get code parameters as a dict."""
        return {
            'type': self.__class__.__name__,
            'codeword_length': self.N,
            'info_length': self.K,
            'pcc_method': self.pcc_method,
            'design_snr': self.design_snr,
            'is_systematic': self.is_systematic,
            'is_crc_aided': self.is_crc_aided,
            'polar_mask': ''.join(str(m) for m in self.polar_mask),
            'extra_params': self._get_extra_params(),
        }

    def encode(self, message):
        """Encode binary message with polar code."""
        if self.is_crc_aided:
            message = self._add_crc(message)

        precoded = self._precode(message)
        encoded = self.non_systematic_encode(precoded)

        if self.is_systematic:
            encoded *= self.polar_mask
            encoded = self.non_systematic_encode(encoded)

        return encoded

    def non_systematic_encode(self, message):
        """Non-systematic encoding.

        Args:
            message (numpy.array): precoded message to encode.

        Returns:
            message (numpy.array): non-systematically encoded message.

        """
        for i in range(self.n - 1, -1, -1):
            message = compute_encoding_step(i, self.n, message, message)

        return message

    def decode(self, received_message):
        """Decode message."""
        raise NotImplementedError()

    @staticmethod
    def _calculate_polar_steps(codeword_length):
        """Calculate number of polar steps `n`."""
        return int(np.log2(codeword_length))

    def compute_channels_estimates(self, pcc_method):
        """"""
        if pcc_method == self.CUSTOM:
            return None

        pcc_method = self.pcc_methods[pcc_method]
        channel_estimates = pcc_method(self.N, self.design_snr)

        # bit-reversal approach https://arxiv.org/abs/1307.7154 (Section III-D)
        for i in range(self.N):
            channel_estimates[i] = channel_estimates[reverse_bits(i, self.n)]

        return channel_estimates

    def polar_code_construction(self, mask=None):
        """Construct polar mask."""
        if mask:
            return np.array([int(b) for b in mask])

        info_length = self.K + self.CRC_SIZE if self.is_crc_aided else self.K
        return self._construct_polar_mask(info_length)

    def _construct_polar_mask(self, info_length):
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
        for m in mask[:info_length]:
            m[2] = 1
        # sort channels due to order
        mask = sorted(mask, key=itemgetter(0))
        # return mask, contains 0s or 1s due to frozen/info position
        return np.array([i[2] for i in mask])

    def _precode(self, info):
        """Apply polar code mask to information message.

        Replace 1's of polar code mask with bits of information message.

        """
        precoded = np.zeros(self.N, dtype=int)
        precoded[self.polar_mask == 1] = info
        return precoded

    def _extract(self, decoded):
        """Extract info bits from decoded message due to polar code mask"""
        decoded_info = np.array(list(), dtype=int)
        mask = self.polar_mask
        for i in range(self.N):
            if mask[i] == 1:
                decoded_info = np.append(decoded_info, decoded[i])
        return decoded_info

    def _add_crc(self, message):
        """Appends CRC 16 value to message."""
        return np.append(message, self._compute_crc_value(message))

    def _compute_crc_value(self, message):
        """Calculate CRC sum of message and add result to message."""
        return int_to_bin_list(calculate_crc_16(message), 16)

    def _check_crc(self, decoded_info):
        """Check decoded information message with CRC algorithm."""
        return check_crc_16(decoded_info)

    def _remove_crc(self, decoded_info):
        """Remove CRC sum from decoded information message."""
        return decoded_info[:self.K]
