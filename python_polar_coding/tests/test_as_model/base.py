import json
import uuid
from unittest import TestCase

import numpy as np

from .channels import SimpleBPSKModAWGNChannel, VerificationChannel


class BasePolarCodeTestMixin:
    """Provides simple BPSK modulator for polar codes testing."""
    messages = None
    codec_class = None
    channel_class = None
    code_parameters = dict()

    #@classmethod
    def setUp(cls):
        cls.codec = cls.codec_class(**cls.code_parameters)
        cls.bit_errors_data = dict()
        cls.frame_errors_data = dict()
        cls.result = cls.codec.to_dict()
        cls.result_path = f'experiments/{str(uuid.uuid4())}'

    @property
    def N(self):
        return self.code_parameters['N']

    @property
    def K(self):
        return self.code_parameters['K']

    def _get_channel(self, snr_db):
        return self.channel_class(snr_db)

    def _message_transmission_test(self, channel, with_noise=True):
        """Basic workflow to compute BER and FER on message transmission"""
        bit_errors = frame_errors = 0  # bit and frame error ratio

        for m in range(self.messages):
            message = np.random.randint(0, 2, self.K)
            encoded = self.codec.encode(message)
            llr = channel.transmit(encoded, with_noise)
            decoded = self.codec.decode(llr)

            fails = np.sum(message != decoded)
            bit_errors += fails
            frame_errors += fails > 0

        return [
            bit_errors / (self.messages * self.K),
            frame_errors / self.messages,
        ]

    def _base_test(self, snr_db=0.0, with_noise=True):
        channel = self._get_channel(snr_db)

        bit_errors, frame_errors = self._message_transmission_test(
            channel,
            with_noise,
        )

        # `-1` means simulation without noise
        snr_db = snr_db if with_noise else '-1'
        self.bit_errors_data.update({str(snr_db): bit_errors})
        self.frame_errors_data.update({str(snr_db): frame_errors})

        return bit_errors, frame_errors

    def test_sc_decoder_without_noise(self):
        """Test a Polar Code without any noisy channel.

        For correctly implemented code the data is transmitted and decoded
        without errors.

        """
        bit_errors, frame_errors = self._base_test(with_noise=False)
        self.assertEqual(bit_errors, 0)
        self.assertEqual(frame_errors, 0)

    def tearDown(self):
        self.result['bit_error_rate'] = self.bit_errors_data
        self.result['frame_error_rate'] = self.frame_errors_data
        self.result['messages'] = self.messages

    @classmethod
    def tearDownClass(cls):
        cls.result['bit_error_rate'] = cls.bit_errors_data
        cls.result['frame_error_rate'] = cls.frame_errors_data
        cls.result['messages'] = cls.messages

        with open(cls._get_filename(), 'w') as fp:
            json.dump(cls.result, fp)

    @classmethod
    def _get_filename(cls):
        N = cls.code_parameters['codeword_length']
        K = cls.code_parameters['info_length']
        return f'{N}_{K}.json'


class SimpleBPSKAWGNTestCase(BasePolarCodeTestMixin, TestCase):
    channel_class = SimpleBPSKModAWGNChannel


class VerificationChannelTestCase(BasePolarCodeTestMixin, TestCase):
    channel_class = VerificationChannel

    def _get_channel(self, snr_db):
        return self.channel_class(snr_db, K=self.K, N=self.N)
