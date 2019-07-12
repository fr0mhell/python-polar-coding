import copy

import numpy as np

import utils

# Boundaries for LLR values that mean definitely ZERO or ONE values
LLR_ZERO, LLR_ONE = 10, -10


class SCBranch:
    """A single branch of SC List decoder or polar code.

    Stores initial and intermediate LLR values, intermediate bit values and
    metrics for forking SC List decoder tree.

    Args:
        received_message (NumPy Array): LLRs of received message.

    Attributes:
        msg_length (int): length of decoding polar code codeword length (N).
        steps (int): number of steps in a given polar code (n, n = log2(N)).
        llrs (NumPy Array): initial and intermediate LLR values for each symbol
            of message to decode. First N-1 are intermediate and recalculated
            each step. Last N are initial values from demodulator and not
            changed in decoding process.
        bits (NumPy Array): 2 x N-1 array of intermediate bit values
        decoded (NumPy Array): decoding results.
        fork_value (float): value (1 or 0) to be selected as a decoded in a
            current branch.
        fork_metric (float): probability that selected fork_value is a correct
            answer. Based on value of 'llrs[0]'.
        correct_prob (float): probability of storing a correct answer in
            'decoded'. Used to select best of given branches.

    """
    def __init__(self, received_message):
        """Successive cancellation branch constructor."""
        self.current_position = 0
        self.msg_length = len(received_message)
        self.steps = int(np.log2(self.msg_length))

        self.llrs = np.append(np.zeros(self.msg_length - 1), received_message)
        self.decoded = np.zeros(self.msg_length, dtype=int)
        self.bits = np.zeros((2, self.msg_length-1), dtype=int)

        # Value of decoded bit set after forking
        # 0 for LLR > 0, 1 for LLR < 0
        self.fork_value = 1
        # Metric that allow to rate `quality` of the branch - probability of
        # correct decoding decision for the current bit
        self.fork_metric = 1
        # Probability of containing the correct decoded data. Used to evaluate
        # the result after all bits decoded
        self.correct_prob = 1

    def update_before_fork(self):
        """Update fork parameters.

        LLR = ln(P0) - ln(P1), P0 + P1 = 1
        exp(LLR) = P0 / P1
        P0 = exp(LLR) / (exp(LLR) + 1)
        P1 = 1 / (exp(LLR) + 1)
        Pi = ( (1 - i) * exp(LLR) + i ) / (exp(LLR) + 1)

        """
        if not (LLR_ONE < self.llrs[0] < LLR_ZERO):
            self.fork_metric = 1
            self.fork_value = 0 if self.llrs[0] > LLR_ZERO else 1
            return
        self.fork_value = 0 if self.llrs[0] > 0 else 1
        if self.llrs[0] == 0:
            self.fork_metric = 0.5
        if self.llrs[0] > 0:
            self.fork_metric = (
                np.exp(self.llrs[0]) / (np.exp(self.llrs[0]) + 1)
            )
        if self.llrs[0] < 0:
            self.fork_metric = (1 / (np.exp(self.llrs[0]) + 1))

    def fork(self):
        """Make a copy of SC branch for List decoding"""
        branch_copy = copy.deepcopy(self)
        branch_copy.fork_metric = 1 - self.fork_metric
        branch_copy.fork_value = 1 - self.fork_value

        return branch_copy

    def __repr__(self):
        return repr((
            self.correct_prob,
            self.fork_metric,
            self.decoded
        ))

    def __lt__(self, other):
        return self.correct_prob < other.correct_prob

    def __gt__(self, other):
        return self.correct_prob > other.correct_prob

    def __eq__(self, other):
        return self.correct_prob == other.correct_prob

    def __le__(self, other):
        return self <= other

    def __ge__(self, other):
        return self >= other

    def __ne__(self, other):
        return self != other

    def update_llrs(self):
        """Update intermediate LLR values."""
        n = self.steps
        position = self.current_position

        if position == 0:
            next_level = n
        else:
            last_level = (bin(position)[2:].zfill(n)).find('1') + 1
            start = int(np.power(2, last_level - 1)) - 1
            end = int(np.power(2, last_level) - 1) - 1
            for i in range(start, end + 1):
                self.llrs[i] = utils.lowerconv(
                    self.bits[0][i],
                    self.llrs[end + 2 * (i - start) + 1],
                    self.llrs[end + 2 * (i - start) + 2]
                )
            next_level = last_level - 1
        for lev in range(next_level, 0, -1):
            start = int(np.power(2, lev - 1)) - 1
            end = int(np.power(2, lev) - 1) - 1
            for i in range(start, end + 1):
                self.llrs[i] = utils.upperconv(
                    self.llrs[end + 2 * (i - start) + 1],
                    self.llrs[end + 2 * (i - start) + 2]
                )

    def update_bits(self):
        """Update intermediate BIT values."""
        N = self.msg_length
        n = self.steps
        position = self.current_position
        last_bit = self.decoded[position]
        if position == N - 1:
            return

        if position < N // 2:
            self.bits[0][0] = last_bit
        else:
            last_level = (bin(position)[2:].zfill(n)).find('0') + 1
            self.bits[1][0] = last_bit
            for lev in range(1, last_level - 1):
                start = int(np.power(2, lev - 1)) - 1
                end = int(np.power(2, lev) - 1) - 1
                for i in range(start, end + 1):
                    self.bits[1][end + 2 * (i - start) + 1] = \
                        (self.bits[0][i] + self.bits[1][i]) % 2
                    self.bits[1][end + 2 * (i - start) + 2] = self.bits[1][i]

            lev = last_level - 1
            start = int(np.power(2, lev - 1)) - 1
            end = int(np.power(2, lev) - 1) - 1
            for i in range(start, end + 1):
                self.bits[0][end + 2 * (i - start) + 1] = \
                    (self.bits[0][i] + self.bits[1][i]) % 2
                self.bits[0][end + 2 * (i - start) + 2] = self.bits[1][i]

    @property
    def can_be_forked(self):
        """Check if the branch can be forked or not.

        Branch can be forked if LLR of currently decoding symbol (self.llrs[0])
        corresponds to:
            LLR_ONE < self.llrs[0] < LLR_ZERO

        Otherwise there is no need to fork this branch, because probability of
        correct result in forked branch is close to zero.

        """
        return LLR_ONE < self.llrs[0] < LLR_ZERO

    @property
    def is_sufficient(self):
        """Check if branch is worth to use further.

        Branches with correct probability less then 0.0001 are insufficient.

        """
        return self.correct_prob >= 0.0001

    def update_correct_probability(self):
        self.correct_prob *= self.fork_metric

    def update_decoding_position(self, pos):
        self.current_position = pos

    def decode_current_bit(self):
        self.decoded[self.current_position] = self.fork_value

    def set_bit_as_frozen(self):
        self.decoded[self.current_position] = 0


def fork_branches(sc_list, max_list_size):
    """Forks SC branches."""
    sc_list_with_forks = list()
    for branch in sc_list:
        # skip insufficient branches
        # if not branch.is_sufficient:
        #     continue
        branch.update_before_fork()
        sc_list_with_forks.append(branch)
        if branch.can_be_forked:
            sc_list_with_forks.append(branch.fork())

    for branch in sc_list_with_forks:
        branch.update_correct_probability()
        branch.decode_current_bit()

    sorted_sc_list = sorted(
        [branch for branch in sc_list_with_forks],
        reverse=True
    )

    if len(sc_list) > max_list_size:
        return [s for s in sorted_sc_list[:max_list_size]]
    return [s for s in sorted_sc_list]
