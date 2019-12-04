import json
from collections import Counter

with open('polar_trees.json') as f:
    code_data = json.load(f)

counters = {
    8: {
        'node_types': Counter(),
        'other_nodes': Counter(),
    },
    16: {
        'node_types': Counter(),
        'other_nodes': Counter(),
    },
    32: {
        'node_types': Counter(),
        'other_nodes': Counter(),
    },
    64: {
        'node_types': Counter(),
        'other_nodes': Counter(),
    },
}

for c in code_data:
    print(f'{c[ "codeword_length"]}, {c["design_snr"]}, '
          f'{c["node_min_size"]}; leaves: {len(c["leaves"])}')

    node_min_size = c['node_min_size']
    leaves = c['leaves']
    leaves_qty = len(c['leaves'])

    count_node_types = Counter([l['type'] for l in leaves])
    count_other_nodes = Counter([l['mask'] for l in leaves if l['type'] == 'OTHER'])

    node_types = counters[node_min_size]['node_types']
    node_types += count_node_types
    counters[node_min_size]['node_types'] = node_types

    other_nodes = counters[node_min_size]['other_nodes']
    other_nodes += count_other_nodes
    counters[node_min_size]['other_nodes'] = other_nodes

for node_min_size, sub_counters in counters.items():
    for k, v in sub_counters.items():
        with open(f'{k}_{node_min_size}.json', 'w') as f:
            json.dump(dict(v), f)
