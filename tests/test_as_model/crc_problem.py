from unittest import TestCase
from .channels import VerificationChannel
from polar_codes import SCListPolarCode
import numpy as np
import json


class TestCRCListDecoding(TestCase):
    """Get data for CRC List decoding investigations."""
    N = 1024
    K = 512
    crc_size = 16
    list_size = 8
    messages = 10
    snr_db = 1.0

    def setUp(self):
        self.crc_code = SCListPolarCode(
            codeword_length=self.N,
            info_length=self.K,
            is_systematic=True,
            list_size=self.list_size,
            is_crc_aided=True,
        )
        self.channel = VerificationChannel(
            snr_db=self.snr_db,
            K=self.K,
            N=self.N,
        )

    def test_crc_investigation(self):
        transmitted = list()

        for m in range(self.messages):
            message = np.random.randint(0, 2, self.K)
            crc_message = self.crc_code._add_crc(message)
            encoded = self.crc_code.encode(message)
            llr = self.channel.transmit(encoded, with_noise=True)
            decoded = self.crc_code.decode(llr)

            transmitted.append({
                'message': message.tolist(),
                'message_with_crc': crc_message.tolist(),
                'decoded': decoded.tolist(),
                'results_with_crc': [
                    self.crc_code._extract(r).tolist()
                    for r in self.crc_code.decoder.result
                ],
            })

        data = {
            'code': self.crc_code.to_dict(),
            'results': transmitted,
        }

        with open('crc_problem.json', 'w') as fp:
            json.dump(data, fp)
