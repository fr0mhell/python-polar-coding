import pprint

import numpy as np


class BasePolarCodeTestMixin:
    """Provides simple BPSK modulator for polar codes testing."""
    messages = None
    codec_class = None
    channel_class = None

    @classmethod
    def setUpClass(cls):
        cls.ber_border = cls.messages // 10
        cls.fer_border = cls.messages // 50
        cls.bit_errors_data = dict()
        cls.frame_errors_data = dict()
        cls.result = dict()

    def _message_transmission_test(self, snr_db, with_noise=False):
        """Basic workflow to compute BER and FER on message transmission"""
        bit_errors = frame_errors = 0  # bit and frame error ratio
        channel = self.channel_class(snr_db)

        for m in range(self.messages):
            message = np.random.randint(0, 2, self.K)
            encoded = self.codec.encode(message)
            llr = channel.transmit(encoded, with_noise)
            decoded = self.codec.decode(llr)

            fails = np.sum(message != decoded)
            bit_errors += fails
            frame_errors += fails > 0

        return [
            {snr_db: bit_errors / (self.messages * self.K)},
            {snr_db: frame_errors / self.messages},
        ]

    def _base_test(self, snr_db=0.0, with_noise=False):
        bit_errors, frame_errors = self._message_transmission_test(
            snr_db,
            with_noise,
        )
        self.bit_errors_data.update(bit_errors)
        self.frame_errors_data.update(frame_errors)
        pprint.pprint(self.bit_errors_data)
        pprint.pprint(self.frame_errors_data)

    @classmethod
    def tearDownClass(cls):
        cls.result.update(cls.codec.to_dict())
        cls.result['bit_error_rate'] = cls.bit_errors_data
        cls.result['frame_error_rate'] = cls.frame_errors_data

        # output of test result
        pprint.pprint(cls.result)
