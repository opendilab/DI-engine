from collections import namedtuple
import torch
import torch.nn as nn
from .policy import Policy
from .encoder import Encoder
from .value import ValueBaseline
from ..actor_critic.actor_critic import ActorCriticBase


class AlphaStarActorCritic(ActorCriticBase):
    EvalOutput = namedtuple('EvalOutput', ['actions', 'next_state'])
    MimicOutput = namedtuple('MimicOutput', ['logits', 'next_state'])
    StepOutput = namedtuple('StepOutput', ['actions', 'baselines', 'next_state_home', 'next_state_away'])

    CriticInput = namedtuple('CriticInput', ['lstm_output_home', 'lstm_output_away', 'baseline_feature_home',
                             'baseline_feature_away', 'cum_stat_home', 'cum_stat_away'])
    CriticOutput = namedtuple('CriticOutput', ['winloss', 'build_orders', 'built_units', 'effects', 'upgrades'])

    def __init__(self, cfg):
        super(AlphaStarActorCritic, self).__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg.encoder)
        self.policy = Policy(cfg.policy)
        if cfg.use_value_network:
            self.value_networks = nn.ModuleDict()
            self.value_cum_stat_keys = {}
            for k, v in cfg.value.items():
                # creating a ValueBaseline network for each baseline, to be used in _critic_forward
                self.value_networks[v.name] = ValueBaseline(v.param)
                # name of needed cumulative stat items
                self.value_cum_stat_keys[v.name] = v.cum_stat_keys

    # overwrite
    def mimic(self, inputs, **kwargs):
        lstm_output, next_state, entity_embeddings, map_skip, scalar_context, _, _ = self.encoder(inputs)
        policy_inputs = self.policy.Input(inputs['actions'], inputs['entity_raw'], lstm_output,
                                          entity_embeddings, map_skip, scalar_context)
        logits = self.policy(policy_inputs, mode='mimic')
        return self.MimicOutput(logits, next_state)

    # overwrite
    def evaluate(self, inputs, **kwargs):
        '''
            Overview: agent evaluate(only actor)
            Note:
                batch size = 1
        '''
        ratio = self.cfg.location_expand_ratio
        Y, X = inputs['map_size'][0]

        lstm_output, next_state, entity_embeddings, map_skip, scalar_context, _, _ = self.encoder(inputs)
        policy_inputs = self.policy.Input(inputs['actions'], inputs['entity_raw'], lstm_output,
                                          entity_embeddings, map_skip, scalar_context)
        actions = self.policy(policy_inputs, mode='evaluate', **kwargs)

        if isinstance(actions['target_location'][0], torch.Tensor):
            location = actions['target_location'][0]
            transformed_location = torch.cat([location // (ratio*X), location % (ratio*X)], 0)
            transformed_location = transformed_location.float().div(ratio)
            actions['target_location'] = [transformed_location]

        # error action(no necessary selected units)
        if isinstance(actions['selected_units'][0], torch.Tensor) and actions['selected_units'][0].shape[0] == 0:
            device = actions['action_type'][0].device
            actions = {'action_type': [torch.LongTensor([0]).to(device)], 'delay': [torch.LongTensor([0]).to(device)],
                       'queued': [None], 'selected_units': [None], 'target_units': [None], 'target_location': [None]}
        return self.EvalOutput(actions, next_state)

    # overwrite
    def step(self, inputs, **kwargs):
        '''
            Overview: agent train(actor and critic)
        '''
        # encoder(home and away)
        lstm_output_home, next_state_home, entity_embeddings, map_skip, scalar_context, baseline_feature_home, cum_stat_home = self.encoder(inputs['home'])  # noqa
        lstm_output_away, next_state_away, _, _, _, baseline_feature_away, cum_stat_away = self.encoder(inputs['away'])

        # value
        critic_inputs = self.CriticInput(lstm_output_home, lstm_output_away, baseline_feature_home,
                                         baseline_feature_away, cum_stat_home, cum_stat_away)
        baselines = self._critic_forward(critic_inputs)

        # policy
        policy_inputs = self.policy.Input(inputs['actions'], inputs['entity_raw'], lstm_output_home,
                                          entity_embeddings, map_skip, scalar_context)
        actions = self.policy(policy_inputs, mode='evaluate', **kwargs)
        return self.StepOutput(actions, baselines, next_state_home, next_state_away)

    # overwrite
    def _critic_forward(self, inputs):
        '''
        Overview: Evaluate value network on each baseline
        '''
        def select_item(data, key):
            # Input: data:dict key:list Returns: ret:list
            # filter data and return a list of values with keys in key
            ret = []
            for k, v in data.items():
                if k in key:
                    ret.append(v)
            return ret
        cum_stat_home, cum_stat_away = inputs['cum_stat_home'], inputs['cum_stat_away']
        # 'lstm_output_home', 'lstm_output_away', 'baseline_feature_home', 'baseline_feature_away'
        # are torch.Tensors and are shared across all baselines
        same_part = torch.cat(inputs[:4], dim=1)
        ret = []
        for (name, m), (_, key) in zip(self.value_networks.items(), self.value_cum_stat_keys.items()):
            cum_stat_home_subset = select_item(cum_stat_home, key)
            cum_stat_away_subset = select_item(cum_stat_away, key)
            inputs = torch.cat([same_part] + cum_stat_home_subset + cum_stat_away_subset, dim=1)
            # apply the value network to inputs
            ret.append(m(inputs))
        return self.CriticOutput(*ret)
