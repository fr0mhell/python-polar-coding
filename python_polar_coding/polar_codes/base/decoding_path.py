from copy import deepcopy

import numpy as np


class DecodingPathMixin:
    """Decoding Path for list decoding."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
        new_path = self.__class__(n=self.n, mask=self.mask,
                                  is_systematic=self.is_systematic)

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
