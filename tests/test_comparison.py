import numpy as np

from polar_codes import SCPolarCode
from polar_codes.decoders import SCDecoder, SCListDecoder
from tests.test_as_model.channels import VerificationChannel

params = {
    'codeword_length': 1024,
    'info_length': 654,
    'is_systematic': True,
}
codec = SCPolarCode(**params)
channel = VerificationChannel(
    1.0,
    K=params['info_length'],
    N=params['codeword_length'],
)
sc = SCDecoder(mask=codec.polar_mask, is_systematic=codec.is_systematic)
mask = codec.polar_mask
sc_path = 0

sc_list = SCListDecoder(
    mask=codec.polar_mask,
    is_systematic=codec.is_systematic,
    list_size=8,
)

message = np.random.randint(0, 2, codec.K)
print('Encoded:')
encoded = codec.encode(message)
print(encoded)
llr = channel.transmit(encoded, with_noise=True)

sc.initialize(llr)
sc_list.initialize(llr)
for i in range(encoded.size):
    print(f'\nPosition {i}\nMask: {mask[i]}')

    sc(i)
    llr = sc.intermediate_llr[-1][0]
    decision = sc._current_decision

    if llr >= 0:
        sc_path -= (llr * decision)
    if llr < 0:
        sc_path += (llr * (1 - decision))

    print('\nSC decoding')
    print(
        f'LLR: {llr}; '
        f'decision: {decision}; '
        f'path_metric: {sc_path}'
    )

    sc_list(i)
    print('\nSC List decoding')
    for path in sc_list.paths:
        print(
            f'LLR: {path.current_llr}; '
            f'decision: {path._current_decision}; '
            f'path_metric: {path._path_metric}\n'
        )
