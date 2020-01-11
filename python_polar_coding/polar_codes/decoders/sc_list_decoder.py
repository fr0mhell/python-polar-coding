from collections import UserList
from copy import deepcopy

import numba
import numpy as np

from ..base.functions import compute_encoding_step
from .sc_decoder import SCDecoder


class ListDecoderPathMixin:
    """Mixin to extend a decoder class to use as a Path in list decoding."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Probability that current path contains correct decoding result
        self._path_metric = 0

    def __eq__(self, other):
        return self._path_metric == other._path_metric

    def __gt__(self, other):
        return self._path_metric > other._path_metric

    def __ge__(self, other):
        return self > other or self == other

    def __lt__(self, other):
        return not (self >= other)

    def __le__(self, other):
        return not (self > other)

    def __str__(self):
        return (
            f'LLR: {self.current_llr}; '
            f'Decision: {self._current_decision}; '
            f'Path metric: {self._path_metric}'
        )

    @property
    def current_llr(self):
        return self.intermediate_llr[-1][0]

    def __deepcopy__(self, memodict={}):
        new_path = self.__class__(self.mask, is_systematic=self.is_systematic)

        # Copy intermediate LLR values
        new_path.intermediate_llr = [
            np.array(llrs) for llrs in self.intermediate_llr
        ]
        # Copy intermediate bit values
        new_path.intermediate_bits = [
            np.array(bits) for bits in self.intermediate_bits
        ]
        # Copy current state
        new_path.current_state = np.array(self.current_state)
        # Copy previous state
        new_path.previous_state = np.array(self.previous_state)

        # Copy path metric
        new_path._path_metric = self._path_metric

        # Make opposite decisions for each path
        self._current_decision = 0
        new_path._current_decision = 1

        return new_path

    def update_path_metric(self):
        """Update path metrics using LLR-based metric.

        Source: https://arxiv.org/abs/1411.7282 Section III-B

        """
        if self.current_llr >= 0:
            self._path_metric -= (self.current_llr * self._current_decision)
        if self.current_llr < 0:
            self._path_metric += (self.current_llr * (1 - self._current_decision))

    def split_path(self):
        """Make a copy of SC path with another decision.

        If LLR of the current position is out of bounds, there is no sense
        of splitting path because LLR >= 20 means 0 and LLR <= -20 means 1.

        """
        new_path = deepcopy(self)
        return [self, new_path]


class SCPath(ListDecoderPathMixin, SCDecoder):
    """A path of a list decoder."""


class SCListDecoder:
    """Class implements SC List decoding.

    Args:
        mask (np.array): Polar code mask.
        is_systematic (bool): Systematic code or not

    """
    def __init__(self, mask, is_systematic=True, list_size=1):
        self.is_systematic = is_systematic
        self.mask = mask
        self.list_size = list_size
        self.paths = [SCPath(mask, is_systematic), ]

    @property
    def L(self):
        return self.list_size

    @property
    def result(self):
        """Decoding result."""
        return [path.result for path in self.paths]

    @property
    def best_result(self):
        """Result from the best path."""
        return self.paths[0].result

    def initialize(self, received_llr):
        """Initialize paths with received message."""
        for path in self.paths:
            path.set_initial_state(received_llr)

    def __call__(self, position, *args, **kwargs):
        """Single step of SC-decoding algorithm to decode one bit."""
        self.set_decoder_state(position)
        self.compute_intermediate_llr(position)

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
            path.set_decoder_state(position)

    def compute_intermediate_llr(self, position):
        """Compute intermediate LLR values of each path."""
        for path in self.paths:
            path.compute_intermediate_alpha(position)

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
            path.compute_intermediate_beta(position)
            path.update_decoder_state()
