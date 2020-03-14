# Python-polar-coding
A package for modelling Polar codes.

## Installation

`pip install git+git://github.com/fr0mhell/python-polar-coding.git#egg=python_polar_coding`

## Current progress

**Polar code construction:**

- [x] Arikan's Bhattacharyya bounds [Section V.A](https://arxiv.org/pdf/1501.02473.pdf);
- [ ] Arikan’s Monte-Carlo estimation [Section V.B](https://arxiv.org/pdf/1501.02473.pdf);
- [ ] Trifonov’s Gaussian approximation [Section V.D](https://arxiv.org/pdf/1501.02473.pdf);

**Decoding:**
- [x] SC Decoding;
- [x] [SC LIST Decoding](https://arxiv.org/abs/1206.0050);
- [ ] [SC STACK Decoding](https://ieeexplore.ieee.org/document/6215306/?denied=);
- [x] [Fast SSC Decoding](https://arxiv.org/abs/1307.7154);
- [x] [RC SCAN Decoding]();
- [x] [Generalized Fast SSC Decoding](https://arxiv.org/pdf/1804.09508.pdf);
- [ ] [Generalized Fast SSC LIST Decoding](https://arxiv.org/pdf/1804.09508.pdf);

**Modulation:**

- [x] BPSK;
- [ ] M-PSK;
- [ ] M-QAM;
