from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from sc2learner.torch_utils import fc_block, build_activation, Transformer, one_hot


class CumulativeStatEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, activation):
        # The order of the concatenation of outputs is the order of input_dim
        super(CumulativeStatEncoder, self).__init__()
        assert isinstance(input_dim, dict)
        self.data = OrderedDict()  # placeholder, to be filled when forward
        self.keys = []
        for k, v in input_dim.items():
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
    transformer -> fc -> squeeze
    B, 20, N1 -> B, 20, N2 -> B, 20, 1 -> B, 20
    '''
    def __init__(self, input_dim, output_dim, activation):
        super(BeginningBuildOrderEncoder, self).__init__()
        self.output_dim = output_dim
        self.act = activation
        self.transformer = Transformer(
            input_dim=input_dim, head_dim=16, hidden_dim=64, output_dim=64, activation=self.act
        )
        self.embedd_fc = fc_block(64, self.output_dim, activation=self.act)

    def _add_seq_info(self, x):
        N = x.shape[1]
        indices = torch.arange(x.shape[1])
        indices_one_hot = one_hot(indices, num=N).to(x.device)
        indices_one_hot = indices_one_hot.unsqueeze(0).repeat(x.shape[0], 1, 1)  # expand to batch dim
        return torch.cat([x, indices_one_hot], dim=2)

    def forward(self, x):
        assert len(x.shape) == 3
        x = self._add_seq_info(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.embedd_fc(x)
        return x


class ScalarEncoder(nn.Module):
    def __init__(self, cfg):
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
        self.keys = []
        self.scalar_context_keys = []
        self.baseline_feature_keys = []

        for k, item in cfg.module.items():
            if item['arch'] == 'fc':
                encoder = fc_block(item['input_dim'], item['output_dim'], activation=self.act)
                setattr(self, k, encoder)
            elif item['arch'] == 'identity':
                setattr(self, k, nn.Identity())
            else:
                if k == 'cumulative_stat':
                    module = CumulativeStatEncoder(
                        input_dim=item['input_dim'], output_dim=item['output_dim'], activation=self.act
                    )
                    setattr(self, k, module)
                elif k == 'beginning_build_order':
                    module = BeginningBuildOrderEncoder(item['input_dim'], item['output_dim'], self.act)
                    setattr(self, k, module)
                else:
                    raise NotImplementedError("key: {}".format(k))
            self.keys.append(k)
            if 'scalar_context' in item.keys() and item['scalar_context']:
                self.scalar_context_keys.append(k)
            if 'baseline_feature' in item.keys() and item['baseline_feature']:
                self.baseline_feature_keys.append(k)

    def forward(self, x):
        '''
        Input: x: a OrderedDict, each key correspond to the input of each encoder module created
                  most modules expect a tensor input of size [batch_size, input_dim]
                  except 'cumulative_stat', which expects a OrderedDict like {'stat1_name': input_tensor1,...}
        Output:
            All output tensors are shaped like [batch_size, feature1_out_dim + feature2_out_dim + ...]
            The order of the concatenation of forward outputs from modules is the order of modules in cfg
            - embedded_scalar: A tensor of all embedded scalar features, see detailed-architecture.txt L91 for more info
            - scalar_context: A tensor of certain scalar features we want to use as context for gating later
            - baseline_feature: A tensor of certain scalar features for baselines
            - cumulative_stat_data: A OrderedDict of computed stats
        '''
        assert (isinstance(x, dict))
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
            else:
                raise ValueError("Required {} not in keys of data: {}.".format(key, x.keys()))
        embedded_scalar = torch.cat(embedded_scalar, dim=1)
        scalar_context = torch.cat(scalar_context, dim=1)
        baseline_feature = torch.cat(baseline_feature, dim=1)
        cum_stat_data = self.cumulative_stat.data if 'cumulative_stat' in self.keys else None
        return embedded_scalar, scalar_context, baseline_feature, cum_stat_data
