import numpy as np


class BPSKModulatorMixin:
    """Provides simple BPSK modulator for polar codes testing."""
    messages = None

    @classmethod
    def setUpClass(cls):
        cls.symbol_energy = 0
        cls.noise_power = 2
        cls.ber_border = cls.messages // 10
        cls.fer_border = cls.messages // 50

    def compute_symbol_energy(self, K, N, snr_db):
        """"""
        return (2 * K / N) * np.power(10, snr_db / 10)

    def transmit_over_bpsk_channel(self, message, N):
        """"""
        modulated = (2 * message - 1) * np.sqrt(self.symbol_energy)
        with_noise = modulated + np.sqrt(self.noise_power / 2) * np.random.randn(N)
        llr = -(4 * np.sqrt(self.symbol_energy) / self.noise_power) * with_noise
        return llr


class BasicPolarDecoderTestMixin:
    """Tests for `SCPolarCode`."""
    input_dataset = None
    output_dataset = None
    code_class = None

    def setUp(self):
        self.common_params = {
            'codeword_length': None,
            'info_length': None,
            'is_systematic': self.is_systematic,
        }

    @property
    def is_systematic(self):
        raise NotImplementedError

    def _check_decoder_on_dataset(self):
        """Check the decoder using the dataset."""
        code = self.code_class(**self.common_params)
        code_name = f'{code.N}, {code.K}'
        input_vectors = self.input_dataset[code_name]
        output_vectors = self.output_dataset[code_name]

        for i, input_vector in enumerate(input_vectors):
            result = code.decode(input_vector)
            self.assertListEqual(list(result), output_vectors[i])
