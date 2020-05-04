import os.path as osp
from collections import namedtuple, OrderedDict
import torch
import torch.nn as nn

from sc2learner.utils import read_config, merge_dicts
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
    CriticOutput = namedtuple(
        'CriticOutput', ['winloss', 'build_order', 'built_unit', 'effect', 'upgrade', 'battle']
    )

    def __init__(self, model_config=None):
        super(AlphaStarActorCritic, self).__init__()
        self.cfg = merge_dicts(alphastar_model_default_config["model"], model_config)
        self.encoder = Encoder(self.cfg.encoder)
        self.policy = Policy(self.cfg.policy)
        if self.cfg.use_value_network:
            self.value_networks = nn.ModuleDict()
            self.value_cum_stat_keys = OrderedDict()
            for k, v in self.cfg.value.items():
                # creating a ValueBaseline network for each baseline, to be used in _critic_forward
                self.value_networks[v.name] = ValueBaseline(v.param)
                # name of needed cumulative stat items
                self.value_cum_stat_keys[v.name] = v.cum_stat_keys
            if not self.cfg.use_battle_reward:
                self.value_networks.pop('battle')
                self.value_cum_stat_keys.pop('battle')

        self.freeze_module(self.cfg.freeze_targets)

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
        lstm_output, next_state, entity_embeddings, map_skip, scalar_context, spatial_info, _, _ = self.encoder(inputs)
        policy_inputs = self.policy.MimicInput(
            inputs['actions'], inputs['entity_raw'], inputs['scalar_info']['available_actions'], lstm_output,
            entity_embeddings, map_skip, scalar_context, spatial_info
        )
        logits = self.policy(policy_inputs, mode='mimic')
        return self.MimicOutput(logits, next_state)

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

        lstm_output, next_state, entity_embeddings, map_skip, scalar_context, spatial_info, _, _ = self.encoder(inputs)
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

        # error action(no necessary selected units)
        if isinstance(actions['selected_units'][0], torch.Tensor) and actions['selected_units'][0].shape[0] == 0:
            device = actions['action_type'][0].device
            actions = {
                'action_type': [torch.LongTensor([0]).to(device)],
                'delay': [torch.LongTensor([0]).to(device)],
                'queued': [None],
                'selected_units': [None],
                'target_units': [None],
                'target_location': [None]
            }
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
        )  # noqa
        lstm_output_away, next_state_away, _, _, _, _, baseline_feature_away, cum_stat_away, \
        score_embedding_away = self.encoder(inputs['away'])  # noqa

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
        ret = []
        for (name_n, m), (name_c, key) in zip(self.value_networks.items(), self.value_cum_stat_keys.items()):
            cum_stat_home_subset = select_item(cum_stat_home, key)
            cum_stat_away_subset = select_item(cum_stat_away, key)
            inputs = torch.cat([same_part] + cum_stat_home_subset + cum_stat_away_subset, dim=1)
            # apply the value network to inputs
            ret.append(m(inputs))
        # if not use_battle_reward, set returns by None
        if not self.cfg.use_battle_reward:
            ret.append(None)
        return self.CriticOutput(*ret)
