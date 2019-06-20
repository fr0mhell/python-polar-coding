import numpy as np
from operator import itemgetter
from utils import (
    calculate_crc_16,
    check_crc_16,
    int_to_bin_list,
)
from .polar_code_construction import bhattacharyya_bounds


class BasicPolarCode:
    """Base Polar code with CRC 16 support.

    Include code construction, encoding and CRC 16 support.

    Args:
        codeword_length (int): Length of codeword (N).
        info_length (int): Length of information part of codeword (K).
        design_snr (float): Value of design Signal-to-Noise ratio for
            polar code construction in dB.
        is_systematic (bool): If systematic encoding used or not.
        is_crc_aided (bool): If CRC used or not.
        dumped_mask (str): Pre-defined Polar mask passed as a string of 0`s and
            1`s.

    """

    def __init__(self, codeword_length, info_length, design_snr,
                 is_systematic=True, is_crc_aided=False, dumped_mask=None):

        assert info_length < codeword_length, (
            f'Cannot create Polar code with N = {codeword_length}, '
            f'K = {info_length}. N must be bigger than K.'
        )

        self.codeword_length = codeword_length
        self.info_length = info_length
        self.design_snr = design_snr
        self.is_systematic = is_systematic
        self.is_crc_aided = is_crc_aided
        self.dumped_mask = dumped_mask

        self.polar_steps = self._calculate_polar_steps(codeword_length)
        self.polar_mask = self._compute_polar_mask()

    @property
    def N(self):
        """Get codeword length using the common name `N`."""
        return self.codeword_length

    @property
    def K(self):
        """Get information word length using the common name `K`."""
        return self.info_length

    @property
    def n(self):
        """Get the number of polar steps using the common name `n`."""
        return self.polar_steps

    def __str__(self):
        return (
            f'({self.N}, {self.K}) Polar code.\n'
            f'Design SNR: {self.design_snr} dB\n'
            f'Systematic: {str(self.is_systematic)}\n'
            f'CRC aided: {str(self.is_crc_aided)}\n'
        )

    def to_dict(self):
        """Get code parameters as a dict."""
        return {
            'codeword_length': self.codeword_length,
            'info_length': self.info_length,
            'design_snr': self.design_snr,
            'is_systematic': self.is_systematic,
            'is_crc_aided': self.is_crc_aided,
            'polar_mask': ''.join(str(m) for m in self.polar_mask)
        }

    def encode(self, message):
        """Encode binary message with polar code."""
        if self.is_crc_aided:
            message = self._add_crc(message)

        precoded = self._precode(message)
        encoded = self._mul_matrix(precoded)

        if self.is_systematic:
            encoded *= self.polar_mask
            encoded = self._mul_matrix(encoded)

        return encoded

    def decode(self, received_message):
        """Decode message."""
        raise NotImplementedError

    def _calculate_polar_steps(self, codeword_length):
        """Calculate number of polar steps `n`."""
        return int(np.log2(codeword_length))

    def _compute_polar_mask(self):
        """Compute polar mask."""
        if self.dumped_mask:
            return self._restore_dumped_mask()

        channel_estimates = bhattacharyya_bounds(self.N, self.design_snr)
        info_length = self.K + 16 if self.is_crc_aided else self.K
        return self._build_polar_mask(info_length, channel_estimates)

    def _restore_dumped_mask(self):
        """Restore polar mask from dump."""
        return np.array([int(b) for b in self.dumped_mask])

    def _build_polar_mask(self, info_length, channel_estimates):
        """Build polar code Mask based on channel estimates.

        0 means frozen bit, 1 means information position.

        """
        # represent each bit as tuple of 3 parameters:
        # (order, channel estimate, frozen / information position)
        mask = [[i, b, 0] for i, b in enumerate(channel_estimates)]
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

    def _mul_matrix(self, message):
        """Multiply message of length N == codeword_length with
        polar code matrix.

        """
        N = self.N
        n = self.n
        for i in range(n):
            if i == 0:
                message[0:N:2] = (message[0:N:2] + message[1:N:2]) % 2
            elif i == (n - 1):
                message[0:int(N/2)] = \
                    (message[0:int(N/2)] + message[int(N/2):N]) % 2
            else:
                enc_step = int(np.power(2, i))
                for j in range(enc_step):
                    message[j:N:(2 * enc_step)] = \
                        (message[j:N:(2 * enc_step)]
                         + message[j + np.power(2, i):N:(2 * enc_step)]) % 2
        return message

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
        return int_to_bin_list(calculate_crc_16(message), 16, True)

    def _check_crc(self, decoded_info):
        """Check decoded information message with CRC algorithm."""
        return check_crc_16(decoded_info)

    def _remove_crc(self, decoded_info):
        """Remove CRC sum from decoded information message."""
        return decoded_info[:self.K]
