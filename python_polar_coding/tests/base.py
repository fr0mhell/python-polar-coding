import numpy as np

from python_polar_coding.channels.simple import SimpleBPSKModulationAWGN


class BasicVerifyPolarCode:
    """Provides simple BPSK modulator for polar codes testing."""
    messages = 1000
    polar_code_class = None
    channel_class = SimpleBPSKModulationAWGN
    code_parameters = None

    @classmethod
    def setUpClass(cls):
        fec_rate = cls.code_parameters['K'] / cls.code_parameters['N']
        cls.channel = cls.channel_class(fec_rate)
        cls.polar_code = cls.polar_code_class(**cls.code_parameters)
        cls.result = cls.polar_code.to_dict()

    @classmethod
    def tearDownClass(cls):
        print(cls.result)

    @property
    def N(self):
        return self.code_parameters['N']

    @property
    def K(self):
        return self.code_parameters['K']

    def test_sc_decoder_without_noise(self):
        """Test a Polar Code without any noise.

        For correctly implemented code the data is transmitted and decoded
        without errors.

        """
        bit_errors, frame_errors = self._message_transmission(
            snr_db=10,
            with_noise=False,
        )
        self.assertEqual(bit_errors, 0)
        self.assertEqual(frame_errors, 0)

        self.result.update({
            'no_noise': {
                'bit_errors': bit_errors,
                'frame_errors': frame_errors
            }
        })

    def test_sc_decoder_10_db(self):
        """Test a Polar Code with low noise power.

        For correctly implemented code the data is transmitted and decoded
        without errors for SNR = 10 dB.

        Use the test as the example of modelling, but without assertions.

        """
        bit_errors, frame_errors = self._modelling_test(snr_db=10.0)
        self.assertEqual(bit_errors, 0)
        self.assertEqual(frame_errors, 0)

    def _get_channel(self):
        fec_rate = self.K / self.N
        return self.channel_class(fec_rate)

    def _message_transmission(self, snr_db, with_noise=True):
        """Basic workflow to compute BER and FER on message transmission"""
        bit_errors = frame_errors = 0

        for m in range(self.messages):
            message = np.random.randint(0, 2, self.K)
            encoded = self.polar_code.encode(message)
            llr = self.channel.transmit(
                message=encoded,
                snr_db=snr_db,
                with_noise=with_noise,
            )
            decoded = self.polar_code.decode(llr)

            fails = np.sum(message != decoded)
            bit_errors += fails
            frame_errors += fails > 0

        return bit_errors, frame_errors

    def _modelling_test(self, snr_db):
        bit_errors, frame_errors = self._message_transmission(
            snr_db=snr_db,
            with_noise=True,
        )
        self.result.update({
            snr_db: {
                'bit_errors': bit_errors,
                'frame_errors': frame_errors
            }
        })
        return bit_errors, frame_errors
