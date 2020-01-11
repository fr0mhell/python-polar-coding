from typing import List, Dict
import numpy as np


class Combination:
    """Combination of polar sub-codes."""

    def __init__(self, samples: Dict, metrics: np.array, mask_types: np.array):
        self._samples = samples
        self._metrics = metrics
        self._mask_types = mask_types

        self._combination_metric = self._compute_metric()

    def _compute_metric(self):
        mask = np.array([self._samples[m] for m in self._mask_types]).flatten()
        return np.sum(mask * self._metrics)

    @property
    def metric(self):
        return self._combination_metric

    def __eq__(self, other):
        return self.metric == other.metric

    def __gt__(self, other):
        return self.metric > other.metric


class PolarCodeGenetic:
    """"""
