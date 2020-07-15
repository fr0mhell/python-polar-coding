import numpy as np

from python_polar_coding.polar_codes.utils import splits


class NodeTypes:
    """Types of decoding nodes."""
    ZERO = 'ZERO'
    ONE = 'ONE'
    SINGLE_PARITY_CHECK = 'SINGLE-PARITY-CHECK'
    REPETITION = 'REPETITION'
    G_REPETITION = 'G-REPETITION'
    RG_PARITY = 'RG-PARITY'

    OTHER = 'OTHER'


class NodeTypeDetector:
    """Class used to detect the type of decoding node."""
    # Minimal size of Single parity check node
    SPC_MIN_SIZE = 4
    # Minimal size of Repetition Fast SSC Node
    REPETITION_MIN_SIZE = 2
    # Minimal number of chunks in generalized nodes
    MIN_CHUNKS = 2

    def __init__(self, *args, **kwargs):
        self.last_chunk_type = None
        self.mask_steps = None

    def __call__(
            self,
            supported_nodes: list,
            mask: np.array,
            AF: int = 0,
    ) -> str:
        """Get type of decoding Node."""
        self.N = mask.size
        self.AF = AF

        if (NodeTypes.ONE in supported_nodes
                and self._is_one(mask)):
            return NodeTypes.ONE
        if (NodeTypes.ZERO in supported_nodes
                and self._is_zero(mask)):
            return NodeTypes.ZERO
        if (NodeTypes.SINGLE_PARITY_CHECK in supported_nodes
                and self._is_single_parity_check(mask)):
            return NodeTypes.SINGLE_PARITY_CHECK
        if (NodeTypes.REPETITION in supported_nodes
                and self._is_repetition(mask)):
            return NodeTypes.REPETITION
        if (NodeTypes.RG_PARITY in supported_nodes
                and self._is_rg_parity(mask)):
            return NodeTypes.RG_PARITY
        if (NodeTypes.G_REPETITION in supported_nodes
                and self._is_g_repetition(mask)):
            return NodeTypes.G_REPETITION

        return NodeTypes.OTHER

    def _is_one(self, mask: np.array) -> bool:
        return np.all(mask == 1)

    def _is_zero(self, mask: np.array) -> bool:
        return np.all(mask == 0)

    def _is_single_parity_check(self, mask: np.array) -> bool:
        return (
            mask.size >= self.SPC_MIN_SIZE and
            mask[0] == 0 and
            np.sum(mask) == mask.size - 1
        )

    def _is_repetition(self, mask: np.array) -> bool:
        return (
            mask.size >= self.REPETITION_MIN_SIZE and
            mask[-1] == 1 and
            np.sum(mask) == 1
        )

    def _is_g_repetition(self, mask: np.array) -> bool:
        """Check the node is Generalized Repetition node.

        Based on: https://arxiv.org/pdf/1804.09508.pdf, Section III, A.

        """
        # 1. Split mask into T chunks, T in range [2, 4, ..., N/2]
        for t in splits(self.MIN_CHUNKS, self.N // 2):
            chunks = np.split(mask, t)

            last = chunks[-1]
            last_ok = self._is_single_parity_check(last) or self._is_one(last)

            if not last_ok:
                continue

            others_ok = all(self._is_zero(c) for c in chunks[:-1])
            if not others_ok:
                continue

            self.last_chunk_type = 1 if self._is_one(last) else 0
            self.mask_steps = t
            return True

        return False

    def _is_rg_parity(self, mask: np.array) -> bool:
        """Check the node is Relaxed Generalized Parity Check node.

        Based on: https://arxiv.org/pdf/1804.09508.pdf, Section III, B.

        """
        # 1. Split mask into T chunks, T in range [2, 4, ..., N/2]
        for t in splits(self.MIN_CHUNKS, self.N // 2):
            chunks = np.split(mask, t)

            first = chunks[0]
            if not self._is_zero(first):
                continue

            ones = 0
            spcs = 0

            for c in chunks[1:]:
                if self._is_one(c):
                    ones += 1
                elif self._is_single_parity_check(c):
                    spcs += 1

            others_ok = (ones + spcs + 1) == t and spcs <= self.AF
            if not others_ok:
                continue

            self.mask_steps = t
            return True

        return False


get_node_type = NodeTypeDetector()
