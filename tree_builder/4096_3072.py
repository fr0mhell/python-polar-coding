import json

from polar_codes import SCPolarCode
from tree_builder.tree_builder import FastSSCTreeBuilder

code = {
    'codeword_length': 4096,
    'info_length': 3072,
    'design_snr': 0.0,
    'node_size': 8,
}

pc = SCPolarCode(**code)
mask = pc.polar_mask
estimates = pc.channel_estimates

tree_params = dict()
tree_params['node_min_size'] = code['node_size']

tree = FastSSCTreeBuilder(
    mask=mask,
    channel_metrics=estimates,
    code_min_size=code['node_size']
)
tree_params['leaves'] = [l.to_dict() for l in tree.leaves]
print(tree_params)

with open('4096_3072_8.json', 'w') as f:
    json.dump(tree_params, f)

code = {
    'codeword_length': 4096,
    'info_length': 3072,
    'design_snr': 0.0,
    'node_size': 16,
}

pc = SCPolarCode(**code)
mask = pc.polar_mask
estimates = pc.channel_estimates

tree_params = dict()
tree_params['node_min_size'] = code['node_size']

tree = FastSSCTreeBuilder(
    mask=mask,
    channel_metrics=estimates,
    code_min_size=code['node_size']
)
tree_params['leaves'] = [l.to_dict() for l in tree.leaves]
print(tree_params)

with open('4096_3072_16.json', 'w') as f:
    json.dump(tree_params, f)
