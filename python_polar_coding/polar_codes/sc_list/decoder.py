import numpy as np

from python_polar_coding.polar_codes.base import BaseDecoder

from .decoding_path import SCPath


class SCListDecoder(BaseDecoder):
    """SC List decoding."""
    path_class = SCPath

    def __init__(self, n: int,
                 mask: np.array,
                 is_systematic: bool = True,
                 L: int = 1):
        super().__init__(n=n, mask=mask, is_systematic=is_systematic)
        self.L = L
        self.paths = [
            self.path_class(n=n, mask=mask, is_systematic=is_systematic),
        ]

    @property
    def result(self):
        """Decoding result."""
        return [path.result for path in self.paths]

    @property
    def best_result(self):
        """Result from the best path."""
        return self.result[0]

    def decode_internal(self, received_llr: np.array) -> np.array:
        """Implementation of SC decoding method."""
        self._set_initial_state(received_llr)

        for pos in range(self.N):
            self._decode_position(pos)

        return self.best_result

    def _set_initial_state(self, received_llr):
        """Initialize paths with received message."""
        for path in self.paths:
            path._set_initial_state(received_llr)

    def _decode_position(self, position):
        """Single step of SC-decoding algorithm to decode one bit."""
        self.set_decoder_state(position)
        self._compute_intermediate_alpha(position)

        if self.mask[position] == 1:
            self._populate_paths()
        if self.mask[position] == 0:
            self.set_frozen_value()

        self._update_paths_metrics()
        self._select_best_paths()
        self._compute_bits(position)

    def set_decoder_state(self, position):
        """Set current state of each path."""
        for path in self.paths:
            path._set_decoder_state(position)

    def _compute_intermediate_alpha(self, position):
        """Compute intermediate LLR values of each path."""
        for path in self.paths:
            path._compute_intermediate_alpha(position)

    def set_frozen_value(self):
        """Set current position to frozen values of each path."""
        for path in self.paths:
            path._current_decision = 0

    def _populate_paths(self):
        """Populate SC paths with alternative decisions."""
        new_paths = list()
        for path in self.paths:
            split_result = path.split_path()
            new_paths += split_result

        self.paths = new_paths

    def _update_paths_metrics(self):
        """Update path metric of each path."""
        for path in self.paths:
            path.update_path_metric()

    def _select_best_paths(self):
        """Select best of populated paths.

        If the number of paths is less then L, all populated paths returned.

        """
        if len(self.paths) <= self.L:
            self.paths = sorted(self.paths, reverse=True)
        else:
            self.paths = sorted(self.paths, reverse=True)[:self.L]

    def _compute_bits(self, position):
        """Compute bits of each path."""
        for path in self.paths:
            path._compute_intermediate_beta(position)
            path._update_decoder_state()
