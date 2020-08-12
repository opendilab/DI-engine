import os.path as osp
from collections import namedtuple, OrderedDict
from functools import reduce
import torch
import torch.nn as nn

from sc2learner.utils import read_config, merge_dicts
from sc2learner.envs import AlphaStarEnv
from .encoder import Encoder
from .policy import Policy
from .value import ValueBaseline
from ..actor_critic.actor_critic import ActorCriticBase

alphastar_model_default_config = read_config(osp.join(osp.dirname(__file__), "actor_critic_default_config.yaml"))


class AlphaStarActorCritic(ActorCriticBase):
    EvalInput = namedtuple(
        'EvalInput', ['map_size', 'entity_raw', 'scalar_info', 'spatial_info', 'entity_info', 'prev_state']
    )
    EvalOutput = namedtuple('EvalOutput', ['actions', 'logits', 'next_state'])
    MimicOutput = namedtuple('MimicOutput', ['logits', 'next_state'])
    StepInput = namedtuple('StepInput', ['home', 'away'])
    StepOutput = namedtuple('StepOutput', ['actions', 'logits', 'baselines', 'next_state_home', 'next_state_away'])
    CriticInput = namedtuple(
        'CriticInput', [
            'lstm_output_home', 'lstm_output_away', 'baseline_feature_home', 'baseline_feature_away',
            'score_embedding_home', 'score_embedding_away', 'cum_stat_home', 'cum_stat_away'
        ]
    )
    CriticOutput = namedtuple('CriticOutput', ['winloss', 'build_order', 'built_unit', 'effect', 'upgrade', 'battle'])

    def __init__(self, model_config=None):
        super(AlphaStarActorCritic, self).__init__()
        cfg = merge_dicts(alphastar_model_default_config["model"], model_config)
        self.cfg = self._merge_input_dim(cfg)
        self.encoder = Encoder(self.cfg.encoder)
        self.policy = Policy(self.cfg.policy)
        if self.cfg.use_value_network:
            self.value_networks = nn.ModuleDict()
            self.value_cum_stat_keys = OrderedDict()
            for k, v in self.cfg.value.items():
                if k in self.cfg.enable_baselines:
                    # creating a ValueBaseline network for each baseline, to be used in _critic_forward
                    self.value_networks[v.name] = ValueBaseline(v.param)
                    # name of needed cumulative stat items
                    self.value_cum_stat_keys[v.name] = v.cum_stat_keys

        self.freeze_module(self.cfg.freeze_targets)

    def _merge_input_dim(self, cfg):
        env_info = AlphaStarEnv({}).info()
        cfg.encoder.obs_encoder.entity_encoder.input_dim = env_info.obs_space['entity'].shape[-1]
        cfg.encoder.obs_encoder.spatial_encoder.input_dim = env_info.obs_space['spatial'].shape[
            0] + cfg.encoder.scatter.output_dim
        handle = cfg.encoder.obs_encoder.scalar_encoder.module
        for k in handle.keys():
            handle[k].input_dim = env_info.obs_space['scalar'].shape[k]
        cfg.encoder.score_cumulative.input_dim = env_info.obs_space['scalar'].shape['score_cumulative']
        return cfg

    def freeze_module(self, freeze_targets=None):
        """
        Note:
            must be called after the model initialization, before the model forward
        """
        if freeze_targets is None:
            # if freeze_targets is not provided, try to use self.freeze_targets
            if self.freeze_targets is None:
                raise Exception("not provided arguments(freeze_targets)")
            else:
                freeze_targets = self.freeze_targets
        else:
            # if freeze_targets is provided, update self.freeze_targets for next usage
            self.freeze_targets = freeze_targets

        def get_submodule(name):
            part = name.split('.')
            module = self
            for p in part:
                module = getattr(module, p)
            return module

        for name in freeze_targets:
            module = get_submodule(name)
            module.eval()
            for m in module.parameters():
                m.requires_grad_(False)

    # overwrite
    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'freeze_targets'):
            self.freeze_module()

    # overwrite
    def mimic(self, inputs, **kwargs):
        lstm_output, next_state, entity_embeddings, map_skip, scalar_context, spatial_info, _, _, _ = self.encoder(
            inputs
        )
        policy_inputs = self.policy.MimicInput(
            inputs['actions'], inputs['entity_raw'], inputs['scalar_info']['available_actions'], lstm_output,
            entity_embeddings, map_skip, scalar_context, spatial_info
        )
        logits = self.policy(policy_inputs, mode='mimic')
        return self.MimicOutput(logits, next_state)

    # overwrite
    def mimic_parallel(self, inputs, **kwargs):
        self.traj = [len(b['spatial_info']) for b in inputs]
        self.batch_size = len(inputs[0]['spatial_info'])
        prev_state = inputs[0].pop('prev_state')
        end_idx = [[i for i in inputs[j]['end_index']] for j in range(len(inputs))]
        inputs = self._merge_traj(inputs)
        # encoder
        embedded_entity, embedded_spatial, embedded_scalar, scalar_context, baseline_feature,\
            cum_stat, entity_embeddings, map_skip = self.encoder.encode_parallel_forward(inputs)
        embedded_entity, embedded_spatial, embedded_scalar = [
            self._split_traj(t) for t in [embedded_entity, embedded_spatial, embedded_scalar]
        ]
        # lstm
        lstm_output = []
        for idx, embedding in enumerate(zip(embedded_entity, embedded_spatial, embedded_scalar)):
            active_state = [i for i in range(self.batch_size) if i not in end_idx[idx]]
            tmp_state = [prev_state[i] for i in active_state]
            tmp_output, tmp_state = self.encoder.core_lstm(embedding[0], embedding[1], embedding[2], tmp_state)
            for _idx, active_idx in enumerate(active_state):
                prev_state[active_idx] = tmp_state[_idx]
            lstm_output.append(tmp_output.squeeze(0))
        next_state = prev_state
        lstm_output = self._merge_traj(lstm_output)
        # head
        policy_inputs = self.policy.MimicInput(
            inputs['actions'], inputs['entity_raw'], inputs['scalar_info']['available_actions'], lstm_output,
            entity_embeddings, map_skip, scalar_context, inputs['spatial_info']
        )
        logits = self.policy(policy_inputs, mode='mimic')
        return self.MimicOutput(logits, next_state)

    def _merge_traj(self, data):
        def merge(t):
            if isinstance(t[0], torch.Tensor):
                # t = torch.stack(t, dim=0)
                # return t.reshape(-1, *t.shape[2:])
                t = torch.cat(t, dim=0)
                return t
            elif isinstance(t[0], list):
                return reduce(lambda x, y: x + y, t)
            elif isinstance(t[0], dict):
                return {k: merge([m[k] for m in t]) for k in t[0].keys()}
            else:
                raise TypeError(type(t[0]))

        if isinstance(data, torch.Tensor):
            return data.reshape(-1, *data.shape[2:])
        else:
            return merge(data)

    def _split_traj(self, data):
        assert isinstance(data, torch.Tensor)
        ret = [d.unsqueeze(0) for d in torch.split(data, self.traj, 0)]
        assert len(ret) == len(self.traj), 'resume data length must equal to original data'
        return ret

    # overwrite
    def evaluate(self, inputs, **kwargs):
        """
            Overview: agent evaluate(only actor)
            Note:
                batch size = 1
            Overview: forward for agent evaluate (only actor is evaluated). batch size must be 1
            Inputs:
                - inputs: EvalInput namedtuple with following fields
                    - map_size
                    - entity_raw
                    - scalar_info
                    - spatial_info
                    - entity_info
                    - prev_state
            Output:
                - EvalOutput named dict
        """
        ratio = self.cfg.policy.location_expand_ratio
        Y, X = inputs['map_size'][0]

        lstm_output, next_state, entity_embeddings, map_skip, scalar_context, spatial_info, _, _, _ = self.encoder(
            inputs
        )
        policy_inputs = self.policy.EvaluateInput(
            inputs['entity_raw'],
            inputs['scalar_info']['available_actions'],
            lstm_output,
            entity_embeddings,
            map_skip,
            scalar_context,
            spatial_info,
        )
        actions, logits = self.policy(policy_inputs, mode='evaluate', **kwargs)

        return self.EvalOutput(actions, logits, next_state)

    # overwrite
    def step(self, inputs, **kwargs):
        """
            Overview: forward for training (actor and critic)
            Inputs:
                - inputs: StepInput namedtuple with observations
                    - away: observation from the rival as EvalInput
                    - home: observation from my self as EvalInput
            Outputs:
                - ret: StepOutput namedtuple containing
                    - actions: output from the model
                    - baselines: critic values
                    - next_state_home
                    - next_state_away
        """
        # encoder(home and away)
        lstm_output_home, \
        next_state_home, \
        entity_embeddings, \
        map_skip, \
        scalar_context, \
        spatial_info, \
        baseline_feature_home, \
        cum_stat_home, \
        score_embedding_home = self.encoder(
            inputs['home']
        )
        lstm_output_away, next_state_away, _, _, _, _, baseline_feature_away, cum_stat_away, \
        score_embedding_away = self.encoder(inputs['away'])

        # value
        critic_inputs = self.CriticInput(
            lstm_output_home, lstm_output_away, baseline_feature_home, baseline_feature_away, score_embedding_home,
            score_embedding_away, cum_stat_home, cum_stat_away
        )
        baselines = self._critic_forward(critic_inputs)

        # policy
        policy_inputs = self.policy.EvaluateInput(
            inputs['home']['entity_raw'], inputs['home']['scalar_info']['available_actions'], lstm_output_home,
            entity_embeddings, map_skip, scalar_context, spatial_info
        )
        actions, logits = self.policy(policy_inputs, mode='evaluate', **kwargs)
        return self.StepOutput(actions, logits, baselines, next_state_home, next_state_away)

    # overwrite
    def _critic_forward(self, inputs):
        """
        Overview: Evaluate value network on each baseline
        """
        def select_item(data, key):
            # Input: data:dict key:list Returns: ret:list
            # filter data and return a list of values with keys in key
            ret = []
            for k, v in data.items():
                if k in key:
                    ret.append(v)
            return ret

        cum_stat_home, cum_stat_away = inputs.cum_stat_home, inputs.cum_stat_away
        # 'lstm_output_home', 'lstm_output_away', 'baseline_feature_home', 'baseline_feature_away'
        # are torch.Tensors and are shared across all baselines
        same_part = torch.cat(inputs[:6], dim=1)
        ret = {k: None for k in self.CriticOutput._fields}
        for (name_n, m), (name_c, key) in zip(self.value_networks.items(), self.value_cum_stat_keys.items()):
            assert name_n == name_c
            cum_stat_home_subset = select_item(cum_stat_home, key)
            cum_stat_away_subset = select_item(cum_stat_away, key)
            inputs = torch.cat([same_part] + cum_stat_home_subset + cum_stat_away_subset, dim=1)
            # apply the value network to inputs
            ret[name_n] = m(inputs)
        return self.CriticOutput(**ret)
