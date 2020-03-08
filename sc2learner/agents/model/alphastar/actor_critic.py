from collections import namedtuple
import torch
from .policy import Policy
from .encoder import Encoder
from .value import ValueBaseline
from ..actor_critic.actor_critic import ActorCriticBase


class AlphaStarActorCritic(ActorCriticBase):
    EvalOutput = namedtuple('EvalOutput', 'actions', 'next_state')
    MimicOutput = namedtuple('MimicOutput', 'logits', 'next_state')
    StepOutput = namedtuple('StepOutput', 'actions', 'baselines', 'next_state_home', 'next_state_away')

    CriticInput = namedtuple('CriticInput', 'lstm_output_home', 'lstm_output_away', 'baseline_feature_home',
                             'baseline_feature_away', 'cum_stat_home', 'cum_stat_away')
    CriticOutput = namedtuple('CriticOutput', 'winloss', 'build_orders', 'built_units', 'effects', 'upgrades')

    def __init__(self, cfg):
        super(AlphaStarActorCritic, self).__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg.encoder)
        self.policy = Policy(cfg.policy)
        if cfg.use_value_network:
            self.value = torch.nn.ModuleDict()
            self.value_cum_stat_keys = {}
            for module in cfg.value:
                self.value[module.name] = ValueBaseline(module.param)
                self.value_cum_stat_keys[module.name] = module.cum_stat_keys

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
        def select_item(data, key):
            ret = []
            for k, v in data.items():
                if k in key:
                    ret.append(v)
            return ret
        cum_stat_home, cum_stat_away = inputs[4:]
        same_part = torch.cat(inputs[:4], dim=1)
        ret = []
        for (name, m), (_, key) in zip(self.value.items(), self.value_cum_stat_keys.items()):
            cum_stat_home_subset = select_item(cum_stat_home, key)
            cum_stat_away_subset = select_item(cum_stat_away, key)
            inputs = torch.cat(same_part + cum_stat_home_subset + cum_stat_away_subset, dim=1)
            ret.append(m(inputs))
        return self.CriticOutput(*ret)
