# Python-polar-coding

A package for Polar codes simulation.

## Installation

```bash
pip install git+git://github.com/fr0mhell/python-polar-coding.git#egg=python_polar_coding
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
design_snr = 2.0
messages = 10000
# SNR in [.0, .5, ..., 4.5, 5]
snr_range = [i / 2 for i in range(11)]

codec = FastSSCPolarCodec(N=N, K=K, design_snr=design_snr)
bpsk = SimpleBPSKModulationAWGN(fec_rate=K/N)

result_ber = dict()
result_fer = dict()

for snr in snr_range:
    ber = 0
    fer = 0

    for _ in range(messages):
        msg = generate_binary_message(size=K)
        encoded = codec.encode(msg)
        transmitted = bpsk.transmit(message=msg, snr_db=snr)
        decoded = codec.decode(transmitted)

        bit_errors, frame_error = compute_fails(msg, decoded)
        ber += bit_errors
        fer += frame_error

    result_ber[snr] = ber
    result_fer[snr] = fer
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

[TODO List](TODO.md)

## License

[MIT License](LICENSE.MD)
