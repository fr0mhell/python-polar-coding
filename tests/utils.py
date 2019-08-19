import numba
import numpy as np


class SimpleBPSKModAWGNChannel:
    """Simple model of BPSK-modulation + AWGN channel.

    Args:
        snr_db (float): Signal-to-noise ratio (dB)

    """
    def __init__(self, snr_db):
        self.snr_db = snr_db
        self.noise_power = 2.0
        self.symbol_energy = self._compute_symbol_energy(
            self.snr_db,
            self.noise_power,
        )

    @staticmethod
    @numba.njit
    def _compute_symbol_energy(snr_db, noise_power):
        snr = np.power(10, snr_db / 10)
        return snr * noise_power

    def transmit(self, message, with_noise=True):
        """Transmit BPSK-modulated message over AWGN message."""
        transmitted = self._modulate(message, self.symbol_energy)
        if with_noise:
            transmitted = self._add_noise(transmitted, self.noise_power)
        return self._llr_detection(
            transmitted,
            self.symbol_energy,
            self.noise_power,
        )

    @staticmethod
    # @numba.njit
    def _modulate(message, symbol_energy):
        """BPSK modulation."""
        return (2 * message - 1) * np.sqrt(symbol_energy)

    @staticmethod
    # @numba.njit
    def _add_noise(signal, noise_power):
        """Add AWGN noise to signal."""
        noise = np.sqrt(noise_power / 2) * np.random.randn(signal.size)
        return signal + noise

    @staticmethod
    # @numba.njit
    def _llr_detection(signal, symbol_energy, noise_power):
        """LLR detection of BPSK signal with AWGN."""
        return -(4 * np.sqrt(symbol_energy) / noise_power) * signal
