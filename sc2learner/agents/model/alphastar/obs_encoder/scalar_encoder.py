import torch
import torch.nn as nn
import torch.nn.functional as F
from sc2learner.nn_utils import fc_block, build_activation


class ScalarEncoder(nn.Module):
    def __init__(self, cfg, template):
        super(ScalarEncoder, self).__init__()
        self.act = build_activation(cfg.activation)
        self.scalar_context_keys = []
        for item in template:
            if 'input_dim' in item.keys() and 'output_dim' in item.keys():
                encoder = fc_block(item['input_dim'], item['output_dim'], activation=self.act)
                setattr(self, item['key'], encoder)
            else:
                pass  # TODO
            if 'scalar_context' in item.keys() and item['scalar_context']:
                self.scalar_context_keys.append(item['key'])

    def forward(self, x):
        assert(isinstance(x, dict))
        embedded_scalar = []
        scalar_context = []
        for k, v in x.items():
            if hasattr(self, k):
                new_v = getattr(self, k)(v)
                embedded_scalar.append(new_v)
                if k in self.scalar_context_keys:
                    scalar_context.append(new_v)
        embedded_scalar = torch.cat(embedded_scalar, dim=1)
        scalar_context = torch.cat(scalar_context, dim=1)
        return embedded_scalar, scalar_context


def transform_scalar_data():
    template = [
        {'key': 'agent_statistics', 'input_dim': 1, 'output_dim': 64, 'other': 'float'},
        {'key': 'race', 'input_dim': 5, 'output_dim': 32, 'scalar_context': True, 'other': 'one-hot 5 value'},
        {'key': 'enemy_race', 'input_dim': 5, 'output_dim': 32, 'scalar_context': True, 'other': 'one-hot 5 value'},  # TODO 10% hidden
        {'key': 'upgrades', 'input_dim': 1, 'output_dim': 128, 'other': 'boolean'},
        {'key': 'enemy_upgrades', 'input_dim': 1, 'output_dim': 128, 'other': 'boolean'},
        {'key': 'time', 'dim': 64, 'other': 'transformer'},  # TODO

        {'key': 'available_actions', 'input_dim': 1, 'output_dim': 64, 'scalar_context': True, 'other': 'boolean vector'},  # TODO
        {'key': 'unit_counts_bow', 'input_dim': 1, 'output_dim': 64, 'other': 'square root'},  # TODO
        {'key': 'mmr', 'input_dim': 6, 'output_dim': 64, 'other': 'min(mmr / 1000, 6)'},
        {'key': 'cumulative_statistics', 'input_dims': [], 'output_dims': [32, 32, 32], 'scalar_context': True, 'other': 'boolean vector, split and concat'},
        {'key': 'beginning_build_order', 'scalar_context': True, 'other': 'transformer'},  # TODO
        {'key': 'last_delay', 'input_dim': 128, 'output_dim': 64, 'other':  'one-hot 128 value'},
        {'key': 'last_action_type', 'input_dims': [], 'output_dims': 128, 'other':  'one-hot xxx value(possible actions number)'},  # TODO
        {'key': 'last_repeat_queued', 'input_dims': [], 'output_dims': 256, 'other':  'one-hot xxx value(possible arguments value numbers)'},  # TODO
    ]
    return template


def test_scalar_encoder():
    class CFG:
        def __init__(self):
            self.activation = 'relu'

    template = transform_scalar_data()
    model = ScalarEncoder(CFG(), template).cuda()
    inputs = {}
    B = 4
    for item in template:
        if 'input_dim' in item.keys() and 'output_dim' in item.keys():
            inputs[item['key']] = torch.randn(B, item['input_dim']).cuda()
    print(model)
    embedded_scalar, scalar_context = model(inputs)
    print(embedded_scalar.shape)
    print(scalar_context.shape)


if __name__ == "__main__":
    test_scalar_encoder()
