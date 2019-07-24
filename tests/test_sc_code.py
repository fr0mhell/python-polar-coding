from unittest import TestCase
from polar_codes import SCPolarCode
import numpy as np
import pprint


class TestSCPolarCode(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.N = 1024
        cls.K = 512
        cls.codec = SCPolarCode(codeword_length=cls.N, info_length=cls.K)
        cls.messages = 1000
        cls.snr_range = np.array([i / 2 for i in range(11)])
        cls.symbol_energy = 1
        cls.noise_power = 2

        cls.messages_border = cls.messages // 10
        cls.fer_border = cls.messages // 50

        cls.result = dict()

    def test_sc_polar_code(self):
        """Not a test exactly, but a model of SC polar code."""
        for snr in self.snr_range:
            ber, fer = 0, 0  # bit and frame error ratio
            self.symbol_energy = (2 * self.K / self.N) * np.power(10, snr / 10)

            for m in range(self.messages):
                message = np.random.randint(0, 2, self.K)
                encoded = self.codec.encode(message)

                transmitted = (2 * encoded - 1) * np.sqrt(self.symbol_energy)

                llr = transmitted + np.sqrt(self.noise_power / 2) * np.random.normal(size=self.N)

                decoded = self.codec.decode(llr)

                fails = np.sum(message != decoded)
                ber += fails
                fer += fails > 0

                if m >= self.messages_border and fer >= self.fer_border:
                    break

            self.result.update({snr: [ber / ((m + 1) * self.N), fer / (m + 1)]})
            pprint.pprint(self.result)


