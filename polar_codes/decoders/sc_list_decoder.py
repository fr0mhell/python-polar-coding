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

        # probability that decoding decision is correct for the current bit
        self.current_decision_metric = 1
        # Probability that current path contains correct decoding result
        self.path_metric = 1

    def __eq__(self, other):
        return self.path_metric == other.path_metric

    def __gt__(self, other):
        return self.path_metric > other.path_metric

    def __ge__(self, other):
        return self > other or self == other

    def __lt__(self, other):
        return not (self >= other)

    def __le__(self, other):
        return not (self > other)

    @staticmethod
    @numba.njit
    def evaluate_current_decision(mask_position, llr, decision):
        """Evaluate probability of the current decision being correct.

        LLR = ln(P0) - ln(P1), P0 + P1 = 1
        exp(LLR) = P0 / P1
        P0 = exp(LLR) / (exp(LLR) + 1)
        P1 = 1 / (exp(LLR) + 1)
        Pi = ( (1 - i) * exp(LLR) + i ) / (exp(LLR) + 1)

        """
        # The decision in frozen position is 100% correct
        if mask_position == 0:
            return 1
        return ((1 - decision) * np.exp(llr) + decision) / (np.exp(llr) + 1)

    def __deepcopy__(self, memodict={}):
        new_path = self.__class__(self.mask, is_systematic=self.is_systematic)
        # Invert the current decision
        new_path.current_decision = (self.current_decision + 1) % 2
        # Invert the current decision metric
        new_path.current_decision_metric = 1 - self.current_decision_metric

        # Update path metrics for both paths
        self.path_metric *= self.current_decision_metric
        new_path.path_metric *= new_path.current_decision_metric

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

        return new_path

    def split_path(self):
        """Make a copy of SC path with another decision."""
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
    def __init__(self, mask, is_systematic=True, list_size=4):
        self.is_systematic = is_systematic
        self.mask = mask
        self.current_decision = 0
        self.list_size = list_size

        path = SCPath(mask, is_systematic)
        self.paths = [path, ]

    def initialize(self, received_llr):
        """Initialize paths with received message."""
        for path in self.paths:
            path.initialize(received_llr)

    def __call__(self, position, *args, **kwargs):
        """Single step of SC-decoding algorithm to decode one bit."""
        self._evaluate_paths(position)

        # Populate and evaluate SC paths when decoding information positions
        if self.mask[position] == 1:
            populated_paths = self._populate_path()
            self.paths = self._select_best_paths(populated_paths)

        self._compute_bits(position)

    @property
    def L(self):
        return self.list_size

    def _evaluate_paths(self, position):
        """Evaluate probability metrics of SC paths."""
        for path in self.paths:
            path.set_decoder_state(position)
            path.compute_intermediate_llr(position)
            path.current_decision = path.make_decision(position)
            path.current_decision_metric = path.evaluate_current_decision(
                self.mask[position],
                path.intermediate_llr[-1][0],
                path.current_decision,
            )

    def _populate_path(self):
        """Populate SC paths with alternative decisions."""
        new_paths = list()
        for path in self.paths:
            new_paths += path.split_path()
        return new_paths

    def _select_best_paths(self, populated_paths):
        """Select best of populated paths.

        If the number of paths is less then L/2, all populated paths returned.

        """
        if len(populated_paths) <= self.L // 2:
            return populated_paths
        return sorted(populated_paths, reverse=True)[:self.L//2]

    def _compute_bits(self, position):
        """Compute bits of each path."""
        for path in self.paths:
            path.compute_intermediate_bits(path.current_decision, position)
            path.update_decoder_state()
