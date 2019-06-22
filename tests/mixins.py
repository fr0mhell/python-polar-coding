class BasicPolarDecoderTestMixin:
    """Tests for `SCPolarCode`."""
    input_dataset = None
    output_dataset = None
    code_class = None

    def setUp(self):
        self.common_params = {
            'codeword_length': None,
            'info_length': None,
            'design_snr': 1.0,
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
            self.assertListEquest(result, output_vectors[i])
