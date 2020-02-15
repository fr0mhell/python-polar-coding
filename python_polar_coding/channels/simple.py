import numba
import numpy as np


class SimpleAWGNChannel:
    """Simple AWGN channel.

    Implemented for the comparison with the SC decoder proposed by H. Vangala,
    E. Viterbo, and Yi Hong (See `PlotPC and PlotPCSystematic`):
    https://ecse.monash.edu/staff/eviterbo/polarcodes.html.

    """
    noise_power = 2.0

    def transmit(self, message: np.array) -> np.array:
        """Transmit BPSK-modulated message over AWGN channel."""
        return self._add_noise(message, self.noise_power)

    @staticmethod
    @numba.njit
    def _add_noise(signal: np.array, noise_power: float) -> np.array:
        """Add AWGN noise to signal."""
        noise = np.sqrt(noise_power / 2) * np.random.randn(signal.size)
        return signal + noise


class SimpleBPSKModulationAWGN:
    """Simple model of BPSK-modulation + AWGN channel.

    Implemented for the comparison with the SC decoder proposed by H. Vangala,
    E. Viterbo, and Yi Hong (See `PlotPC and PlotPCSystematic`):
    https://ecse.monash.edu/staff/eviterbo/polarcodes.html.

    """
    noise_power = 2.0

    def __init__(self, fec_rate: float):
        self.fec_rate = fec_rate

    def transmit(self, message: np.array,
                 snr_db: float,
                 with_noise: bool = True) -> np.array:
        """Transmit BPSK-modulated message over AWGN message."""
        symbol_energy = self._compute_symbol_energy(snr_db, self.fec_rate)
        transmitted = self._modulate(message, symbol_energy)

        if with_noise:
            transmitted = self._add_noise(transmitted, self.noise_power)

        return self._llr_detection(transmitted, symbol_energy, self.noise_power)  # noqa

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
    def _add_noise(signal: np.array, noise_power: float) -> np.array:
        """Add AWGN noise to signal."""
        noise = np.sqrt(noise_power / 2) * np.random.randn(signal.size)
        return signal + noise

    @staticmethod
    @numba.njit
    def _llr_detection(signal: np.array, symbol_energy: float, noise_power: float) -> np.array:
        """LLR detection of BPSK signal with AWGN."""
        return -(4 * np.sqrt(symbol_energy) / noise_power) * signal
