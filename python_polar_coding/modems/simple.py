import numba
import numpy as np


class SimpleBPSKModem:
    """Simple model of BPSK-modem.

    Implemented for the comparison with the SC decoder proposed by H. Vangala,
    E. Viterbo, and Yi Hong (See `PlotPC and PlotPCSystematic`):
    https://ecse.monash.edu/staff/eviterbo/polarcodes.html.

    """
    noise_power = 2.0

    def __init__(self, fec_rate: float, snr_db: float):
        self.fec_rate = fec_rate
        self.symbol_energy = self._compute_symbol_energy(snr_db, self.fec_rate)

    def modulate(self, message: np.array) -> np.array:
        """BPSK modulation."""
        return self._modulate(message, self.symbol_energy)

    def demodulate(self, transmitted: np.array) -> np.array:
        """BPSK demodulation."""
        return self._llr_detect(transmitted, self.symbol_energy, self.noise_power)  # noqa

    @staticmethod
    @numba.njit
    def _compute_symbol_energy(snr_db, fec_rate):
        snr = np.power(10, snr_db / 10)
        return snr * 2 * fec_rate

    @staticmethod
    @numba.njit
    def _modulate(message: np.array, symbol_energy: float) -> np.array:
        """BPSK modulation."""
        return (2 * message - 1) * np.sqrt(symbol_energy)

    @staticmethod
    @numba.njit
    def _llr_detect(signal: np.array, symbol_energy: float, noise_power: float) -> np.array:
        """LLR detection of BPSK signal with AWGN."""
        return -(4 * np.sqrt(symbol_energy) / noise_power) * signal
