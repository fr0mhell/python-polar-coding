import pprint
from unittest import TestCase
from .mixins import BPSKModulatorMixin

import numpy as np

from polar_codes import SCPolarCode


class TestSCPolarCode(BPSKModulatorMixin, TestCase):
    messages = 10000

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.N = 64
        cls.K = 45
        cls.codec = SCPolarCode(codeword_length=cls.N, info_length=cls.K)
        cls.snr_range = np.array([i / 2 for i in range(11)])
        cls.bit_errors_data = dict()
        cls.frame_errors_data = dict()
        cls.result = dict()

    def _message_transmission_test(self, snr_db):
        """Basic workflow to compute BER and FER on message transmission"""
        ber = fer = 0  # bit and frame error ratio
        self.symbol_energy = self.compute_symbol_energy(self.K, self.N, snr_db)

        for m in range(self.messages):
            message = np.random.randint(0, 2, self.K)
            encoded = self.codec.encode(message)
            llr = self.transmit_over_bpsk_channel(encoded, self.N)
            decoded = self.codec.decode(llr)

            fails = np.sum(message != decoded)
            ber += fails
            fer += fails > 0

            if m >= self.ber_border and fer >= self.fer_border:
                break

        return [
            {snr_db: ber / ((m + 1) * self.K)},
            {snr_db: fer / (m + 1)},
        ]

    def _base_test(self, snr_db=0.0):
        bit_result, frame_result = self._message_transmission_test(snr_db)
        self.bit_errors_data.update(bit_result)
        self.frame_errors_data.update(frame_result)

    def test_sc_decoder_without_noise(self):
        self._base_test()

    def test_snr_0_0_db(self):
        snd_db = 0.0
        # self._base_test(snd_db)

    def test_snr_0_5_db(self):
        snd_db = 0.5
        # self._base_test(snd_db)

    def test_snr_1_0_db(self):
        snd_db = 1.0
        # self._base_test(snd_db)

    def test_snr_1_5_db(self):
        snd_db = 1.5
        # self._base_test(snd_db)

    def test_snr_2_0_db(self):
        snd_db = 2.0
        # self._base_test(snd_db)

    def test_snr_2_5_db(self):
        snd_db = 2.5
        # self._base_test(snd_db)

    def test_snr_3_0_db(self):
        snd_db = 3.0
        # self._base_test(snd_db)

    def test_snr_3_5_db(self):
        snd_db = 3.5
        # self._base_test(snd_db)

    def test_snr_4_0_db(self):
        snd_db = 4.0
        # self._base_test(snd_db)

    def test_snr_4_5_db(self):
        snd_db = 4.5
        # self._base_test(snd_db)

    def test_snr_5_0_db(self):
        snd_db = 5.0
        # self._base_test(snd_db)

    def tearDown(self):
        pprint.pprint(self.result)

    @classmethod
    def tearDownClass(cls):
        cls.result.update(cls.codec.to_dict())
        cls.result['bit_error_rate'] = cls.bit_errors_data
        cls.result['frame_error_rate'] = cls.frame_errors_data

        # output of test result
        pprint.pprint(cls.result)
