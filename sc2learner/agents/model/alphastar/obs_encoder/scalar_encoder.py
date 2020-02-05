import torch
import torch.nn as nn
import torch.nn.functional as F
from sc2learner.nn_utils import fc_block, build_activation, Transformer
from sc2learner.envs.observations.alphastar_obs_wrapper import transform_scalar_data


class CumulativeStatEncoder(nn.Module):
    def __init__(self, input_dims, output_dim, activation):
        super(CumulativeStatEncoder, self).__init__()
        for k, v in input_dims.items():
            module = fc_block(v, output_dim, activation=activation)
            setattr(self, k, module)

    def forward(self, x):
        ret = []
        for k, v in x.items():
            ret.append(getattr(self, k)(v))
        return torch.cat(ret, dim=1)


class BeginningBuildOrderEncoder(nn.Module):
    def __init__(self, input_dim, begin_num, output_dim, activation):
        super(BeginningBuildOrderEncoder, self).__init__()
        self.act = activation
        self.fc1 = fc_block(input_dim, 16, activation=self.act)
        self.fc2 = fc_block(16, 1, activation=self.act)
        self.transformer = Transformer(
            input_dim=begin_num,
            head_dim=output_dim // 2,
            hidden_dim=output_dim,
            output_dim=output_dim,
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.squeeze(2)
        x = self.transformer(x)
        return x


class ScalarEncoder(nn.Module):
    def __init__(self, cfg, template=None):
        super(ScalarEncoder, self).__init__()
        self.act = build_activation(cfg.activation)
        self.use_stat = cfg.use_stat
        self.scalar_context_keys = []
        if template is None:
            template_obs, template_replay, template_act = transform_scalar_data()
            template = template_obs + template_act
            if self.use_stat:
                template += template_replay
        for item in template:
            if item['arch'] == 'fc':
                encoder = fc_block(item['input_dim'], item['output_dim'], activation=self.act)
                setattr(self, item['key'], encoder)
            else:
                key = item['key']
                if key == 'time':
                    time_transformer = Transformer(
                        input_dim=item['input_dim'],
                        head_dim=item['output_dim'] // 2,
                        hidden_dim=item['output_dim'],
                        output_dim=item['output_dim'])
                    setattr(self, key, time_transformer)
                elif key == 'cumulative_stat':
                    module = CumulativeStatEncoder(
                        input_dims=item['input_dims'],
                        output_dim=item['output_dim'],
                        activation=self.act
                    )
                    setattr(self, key, module)
                elif key == 'beginning_build_order':
                    module = BeginningBuildOrderEncoder(item['input_dim'], cfg.begin_num, item['output_dim'], self.act)
                    setattr(self, key, module)
                else:
                    raise NotImplementedError("key: {}".format(key))
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


def test_scalar_encoder():
    class CFG:
        def __init__(self):
            self.activation = 'relu'

    template_obs = transform_scalar_data()[0]
    model = ScalarEncoder(CFG(), template_obs).cuda()
    inputs = {}
    B = 4
    for item in template_obs:
        if 'input_dim' in item.keys() and 'output_dim' in item.keys():
            inputs[item['key']] = torch.randn(B, item['input_dim']).cuda()
    print(model)
    embedded_scalar, scalar_context = model(inputs)
    print(embedded_scalar.shape)
    print(scalar_context.shape)


if __name__ == "__main__":
    test_scalar_encoder()
