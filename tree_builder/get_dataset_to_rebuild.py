import numpy as np

node_size = 8

ZERO_NODE = np.zeros(node_size, dtype=np.int8)
REPETITION = np.array([1 if i == node_size - 1 else 0 for i in range(node_size)], dtype=np.int8)
SINGLE_PARITY = np.array([0 if i == 0 else 1 for i in range(node_size)], dtype=np.int8)
ONE_NODE = np.ones(node_size, dtype=np.int8)

fast_ssc_nodes_examples = [
    ONE_NODE,
    SINGLE_PARITY,
    REPETITION,
    ZERO_NODE,
]


def nodes_into_two_groups(nodes):
    other_nodes = []
    fast_ssc_nodes = []

    for n in nodes:
        if n['type'].lower() == 'other':
            other_nodes.append(n)
        else:
            fast_ssc_nodes.append(n)

    fast_ssc_nodes = sorted(fast_ssc_nodes, key=lambda node: np.sum(node['mask'] * node['metrics']))

    return fast_ssc_nodes, other_nodes


def check_other_nodes_can_be_rebuilt_to_fast_ssc(nodes, fast_ssc_nodes):
    """"""
    T = len(nodes)
    K = 0
    for node in nodes:
        K += np.sum(node['mask'])

    for node in fast_ssc_nodes:
        if len(node) < T / K:
            return False

        T = T - K // len(node)
        K = K % len(node)

    return True


def get_nodes_to_rebuild(fast_ssc_nodes, other_nodes):
    """"""
    while True:
        if check_other_nodes_can_be_rebuilt_to_fast_ssc(other_nodes, fast_ssc_nodes_examples):
            return other_nodes

        other_nodes.append(fast_ssc_nodes.pop(0))


def get_combinations(combinations, current_cap, nxt, T, K):
    """"""
