from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from sc2learner.torch_utils import fc_block, build_activation, Transformer
from sc2learner.envs.observations.alphastar_obs_wrapper import transform_scalar_data


class CumulativeStatEncoder(nn.Module):
    def __init__(self, input_dims, output_dim, activation):
        # The order of the concatenation of outputs is the order of input_dims
        super(CumulativeStatEncoder, self).__init__()
        self.keys = []
        for k, v in input_dims.items():
            module = fc_block(v, output_dim, activation=activation)
            setattr(self, k, module)
            self.keys.append(k)

    def forward(self, x):
        ret = OrderedDict()
        for k in self.keys:
            if k in x.keys():
                ret[k] = getattr(self, k)(x[k])
        self.data = ret
        return torch.cat(list(ret.values()), dim=1)


class BeginningBuildOrderEncoder(nn.Module):
    '''
    Overview:
    x -> fc -> fc -> transformer -> out
    '''
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
        '''
        Overview: Build a ScalarEncoder according to the given template or return of  transform_scalar_data
        template is a list of dicts describing each module of the encoder
        The order of the concatenation of forward outputs from modules is the order of modules in template
        Created modules will be set as attributes of the ScalarEncoder
        the keys used in the dicts:
            - key: the name of the encoder module and the attribute created for the module
            - arch: if set to 'fc', a fc block will be created
            - input_dim
            - output_dim
            - input_dims: cumulative_stat only, OrderedDict describing the stats to be computed and their input_dim
            - scalar_context: set this key to True to label the module should be included in scalar_context
            - baseline_feature: set this key to True to label the module should be included in baseline_feature
        '''
        super(ScalarEncoder, self).__init__()
        self.act = build_activation(cfg.activation)
        self.use_stat = cfg.use_stat
        self.keys = []
        self.scalar_context_keys = []
        self.baseline_feature_keys = []
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
            self.keys.append(item['key'])
            if 'scalar_context' in item.keys() and item['scalar_context']:
                self.scalar_context_keys.append(item['key'])
            if 'baseline_feature' in item.keys() and item['baseline_feature']:
                self.baseline_feature_keys.append(item['key'])

    def forward(self, x):
        '''
        Input: x: a OrderedDict, each key correspond to the input of each encoder module created
                  most modules expect a tensor input of size [batch_size, input_dim]
                  except 'cumulative_stat', which expects a OrderedDict like {'stat1_name': input_tensor1,...}
        Output:
            All output tensors are shaped like [batch_size, feature1_out_dim + feature2_out_dim + ...]
            The order of the concatenation of forward outputs from modules is the order of modules in template
            - embedded_scalar: A tensor of all embedded scalar features, see detailed-architecture.txt L91 for more info
            - scalar_context: A tensor of certain scalar features we want to use as context for gating later
            - baseline_feature: A tensor of certain scalar features for baselines
            - cumulative_stat_data: A OrderedDict of computed stats
        '''
        assert(isinstance(x, dict))
        embedded_scalar = []
        scalar_context = []
        baseline_feature = []
        for key in self.keys:
            if key in x.keys():
                embedding = getattr(self, key)(x[key])
                embedded_scalar.append(embedding)
                if key in self.scalar_context_keys:
                    scalar_context.append(embedding)
                if key in self.baseline_feature_keys:
                    baseline_feature.append(embedding)
        embedded_scalar = torch.cat(embedded_scalar, dim=1)
        scalar_context = torch.cat(scalar_context, dim=1)
        baseline_feature = torch.cat(baseline_feature, dim=1)
        return embedded_scalar, scalar_context, baseline_feature, self.cumulative_stat.data


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
    embedded_scalar, scalar_context, baseline_feature, cumulative_stat = model(inputs)
    print(embedded_scalar.shape)
    print(scalar_context.shape)
    print(baseline_feature.shape)


if __name__ == "__main__":
    test_scalar_encoder()
