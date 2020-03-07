import torch
from .policy import Policy
from .encoder import Encoder
from ..actor_critic.actor_critic import ActorCriticBase


class AlphaStarActorCritic(ActorCriticBase):
    def __init__(self, cfg):
        super(AlphaStarActorCritic, self).__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.policy = Policy(cfg)
        if cfg.use_value_network:
            raise NotImplementedError

    # overwrite
    def mimic(self, inputs, **kwargs):
        lstm_output, next_state, entity_embeddings, map_skip, scalar_context, _ = self.encoder(inputs)
        inputs['lstm_output'], inputs['entity_embeddings'], inputs['map_skip'], inputs['scalar_context'] = lstm_output, entity_embeddings, map_skip, scalar_context  # noqa
        logits = self.policy(inputs, mode='mimic')
        return logits, next_state

    # overwrite
    def evaluate(self, inputs, **kwargs):
        '''
            Overview: agent evaluate(only actor)
            Note:
                batch size = 1
        '''
        ratio = self.cfg.location_expand_ratio
        Y, X = inputs['map_size'][0]

        lstm_output, next_state, entity_embeddings, map_skip, scalar_context, _ = self.encoder(inputs)
        inputs['lstm_output'], inputs['entity_embeddings'], inputs['map_skip'], inputs['scalar_context'] = lstm_output, entity_embeddings, map_skip, scalar_context  # noqa
        actions = self.policy(inputs, mode='evaluate', **kwargs)

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
        return {
            'actions': actions,
            'next_state': next_state
        }
