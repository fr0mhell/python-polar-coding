import numba
import numpy as np


class SimpleBPSKModulationAWGN:
    """Simple model of BPSK-modulation + AWGN channel."""
    symbol_energy = 1.0

    def __init__(self, fec_rate: float):
        self.fec_rate = fec_rate

    def transmit(self, message: np.array, snr_db: float,
                 with_noise: bool = True) -> np.array:
        """Transmit BPSK-modulated message over AWGN message."""
        transmitted = self._modulate(message, self.symbol_energy)
        noise_power = self._compute_noise_power(snr_db, self.fec_rate)

        if with_noise:
            transmitted = self._add_noise(transmitted, noise_power)

        return self._llr_detection(transmitted, self.symbol_energy, noise_power)

    def _compute_noise_power(self, snr_db: float, fec_rate: float) -> float:
        return self.symbol_energy / (2 * fec_rate * np.power(10, snr_db / 10))

    @staticmethod
    @numba.njit
    def _modulate(message: np.array, symbol_energy: float) -> np.array:
        """BPSK modulation."""
        return (2 * message - 1) * np.sqrt(symbol_energy)

    @staticmethod
    @numba.njit
    def _add_noise(signal: np.array, noise_power: float) -> np.array:
        """Add AWGN noise to signal."""
        noise = np.sqrt(noise_power / 2) * np.random.randn(signal.size)
        return signal + noise

    @staticmethod
    @numba.njit
    def _llr_detection(signal: np.array, symbol_energy: float, noise_power: float) -> np.array:
        """LLR detection of BPSK signal with AWGN."""
        return -(4 * np.sqrt(symbol_energy) / noise_power) * signal
