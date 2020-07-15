from typing import Dict, Tuple

import numpy as np
from anytree import Node

from .functions import (
    NodeTypes,
    compute_beta_hard,
    compute_beta_soft,
    get_node_type,
)


class BaseDecodingNode(Node):
    """Base class for node of the decoding tree."""

    LEFT = 'left_child'
    RIGHT = 'right_child'
    ROOT = 'root'
    NODE_NAMES: Tuple = (LEFT, RIGHT, ROOT)

    # Supported types of decoding nodes
    supported_nodes: Tuple = None

    def __init__(self, mask: np.array, name: str = ROOT, AF: int = 0, **kwargs):  # noqa
        """A node of Fast SSC decoder."""
        if name not in self.__class__.NODE_NAMES:
            raise ValueError('Wrong Fast SSC Node type')

        super().__init__(name, **kwargs)

        self.mask = mask
        self.AF = AF
        self.node_type = get_node_type(
            supported_nodes=self.supported_nodes,
            mask=self.mask,
            AF=self.AF,
        )
        self.is_computed = False

        self._alpha = np.zeros(self.N, dtype=np.double)
        self._beta = np.zeros(self.N, dtype=np.int8)

        # For generalized decoders
        self.last_chunk_type = get_node_type.last_chunk_type
        self.mask_steps = get_node_type.mask_steps

        self.build_decoding_tree()

    def __str__(self):
        return ''.join([str(m) for m in self.mask])

    def __len__(self):
        return self.mask.size

    @property
    def N(self) -> int:
        return self.mask.size

    @property
    def alpha(self) -> np.array:
        return self._alpha

    @alpha.setter
    def alpha(self, value: np.array):
        if self.mask.size != value.size:
            raise ValueError('Wrong size of LLR vector')
        self._alpha = np.array(value)

    @property
    def beta(self) -> np.array:
        return self._beta

    @beta.setter
    def beta(self, value: np.array):
        if self.mask.size != value.size:
            raise ValueError('Wrong size of Bits vector')
        self._beta = np.array(value)

    @property
    def is_left(self) -> bool:
        return self.name == self.__class__.LEFT

    @property
    def is_right(self) -> bool:
        return self.name == self.__class__.RIGHT

    @property
    def is_simplified_node(self) -> bool:
        return self.node_type != NodeTypes.OTHER

    def to_dict(self) -> Dict:
        return {
            'type': self.node_type,
            'mask': self.mask,
        }

    def build_decoding_tree(self):
        """Build decoding tree."""
        if self.is_simplified_node:
            return

        left_mask, right_mask = np.split(self.mask, 2)
        cls = self.__class__
        cls(mask=left_mask, name=self.LEFT, AF=self.AF, parent=self)
        cls(mask=right_mask, name=self.RIGHT, AF=self.AF, parent=self)

    def get_decoding_params(self) -> Dict:
        """Get decoding params to perform the decoding in a leaf node."""
        raise NotImplementedError('Implement in a concrete class')

    def __call__(self, *args, **kwargs):
        """Compute beta value of the decoding node."""
        raise NotImplementedError('Implement in a concrete class')


class HardNode(BaseDecodingNode):
    """Decoding node for hard decoding methods."""

    def __call__(self, *args, **kwargs):
        """Compute beta value of the decoding node."""

        if not self.is_leaf:
            raise TypeError('Cannot make decision in not a leaf node.')

        params = self.get_decoding_params()
        self.beta = compute_beta_hard(**params)


class SoftNode(BaseDecodingNode):
    """Decoding node for soft decoding methods."""

    def __call__(self, *args, **kwargs):
        """Compute beta value of the decoding node."""

        if not self.is_leaf:
            raise TypeError('Cannot make decision in not a leaf node.')

        params = self.get_decoding_params()
        self.beta = compute_beta_soft(**params)
