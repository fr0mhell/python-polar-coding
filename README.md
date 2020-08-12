# Python-polar-coding

A package for Polar codes simulation.

## Installation

```bash
pip install python-polar-coding
```

## Example

Here is a simple example of simulation using `python_polar_coding`.

Binary messages encoded with Polar code, modulated using BPSK, transmitted over
channel with AWGN and decoded using [Fast SSC](https://arxiv.org/abs/1307.7154) algorithm.

```python
from python_polar_coding.channels import SimpleBPSKModulationAWGN
from python_polar_coding.polar_codes import FastSSCPolarCodec
from python_polar_coding.simulation.functions import (
    compute_fails,
    generate_binary_message,
)

N = 128
K = 64
design_snr = 0.0
messages = 1000
# SNR in [.0, .5, ..., 4.5, 5]
snr_range = [i / 2 for i in range(11)]

codec = FastSSCPolarCodec(N=N, K=K, design_snr=design_snr)
bpsk = SimpleBPSKModulationAWGN(fec_rate=K/N)

result_ber = dict()
result_fer = dict()

print('Python polar coding simulation')
print(f'Simulating ({codec.N}, {codec.K}) systematic polar code with Design SNR {codec.design_snr} dB')
print()
print('\tSNR (dB)|\tBER\t|\tFER')

for snr in snr_range:
    ber = 0
    fer = 0

    for _ in range(messages):
        msg = generate_binary_message(size=K)
        encoded = codec.encode(msg)
        transmitted = bpsk.transmit(message=encoded, snr_db=snr)
        decoded = codec.decode(transmitted)

        bit_errors, frame_error = compute_fails(msg, decoded)
        ber += bit_errors
        fer += frame_error

    result_ber[snr] = ber / (messages * codec.K)
    result_fer[snr] = fer / messages

    print(f'\t{snr}\t|\t{result_ber[snr]:.4f}\t|\t{result_fer[snr]:.4f}')
```

## Current progress

### Polar code construction

- [x] Arikan's Bhattacharyya bounds [Section V.A](https://arxiv.org/pdf/1501.02473.pdf)

### Decoding
- [x] SC Decoding
- [x] [SC LIST Decoding](https://arxiv.org/abs/1206.0050)
- [x] [Fast SSC Decoding](https://arxiv.org/abs/1307.7154)
- [x] [RC SCAN Decoding]()
- [x] [Generalized Fast SSC Decoding](https://arxiv.org/pdf/1804.09508.pdf)

### Modulation

- [x] BPSK

## TODO

### Polar code construction

- [ ] Arikan’s Monte-Carlo estimation [Section V.B](https://arxiv.org/pdf/1501.02473.pdf)
- [ ] Trifonov’s Gaussian approximation [Section V.D](https://arxiv.org/pdf/1501.02473.pdf)

### Decoding
- [ ] [SC STACK Decoding](https://ieeexplore.ieee.org/document/6215306)
- [ ] [Fast SSC List Decoding](https://arxiv.org/pdf/1703.08208.pdf)
- [ ] [Generalized Fast SSC LIST Decoding](https://arxiv.org/pdf/1804.09508.pdf)
- [ ] CRC-aided decoders

### Modulation

- [ ] Q-PSK
- [ ] 4-QAM

## License

[MIT License](LICENSE.txt)
