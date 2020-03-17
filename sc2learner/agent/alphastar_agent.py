"""
Copyright 2020 Sensetime X-lab. All Rights Reserved

This is a temporary wrapper class Agent. This class wraps the torch model and provides useful API likes compute_actions,
and so on. Besides, it also maintain the previous state of RNN like models.
"""

import logging
from collections import namedtuple

import torch

from sc2learner.torch_utils import to_device
from sc2learner.utils import DistModule

# TODO(pzh) I think actions should be `action` and logits should be `logit` to be consistent with `next_state` ...
AgentOutput = namedtuple("AgentOutput", ["actions", "logits", "next_state"])

PREV_STATE = "prev_state"


class NullContext:
    def __enter__(self):
        pass

    def __exit__(self, *_):
        pass


class BaseAgent:
    """ To be done """
    pass


class AlphaStarAgent(BaseAgent):

    def __init__(self, cfg, build_model, use_cuda, use_distributed):
        self.cfg = cfg
        self.num_concurrent_episodes = cfg.data.train.batch_size

        # Build model
        self.model = build_model(cfg)
        self.model.train()  # set model to train
        self.use_cuda = use_cuda
        if use_cuda:
            self.model = to_device(self.model, "cuda")
        if use_distributed:  # distributed training
            self.model = DistModule(self.model)

        self.episode_ids = set(range(self.num_concurrent_episodes))
        self.prev_state = {episode_id: None for episode_id in self.episode_ids}
        self.active_episodes = {episode_id: False for episode_id in self.episode_ids}

    @property
    def is_training(self):
        # TODO(pzh): a workaround to deal with mimic mode and evaluate mode
        return self.model.training

    def reset_previous_state(self, if_new_episodes):
        """ Call this function when a batch of data start

        if_new_episodes is a boolean list eauql to batch[0][start_step]
        """

        if len(if_new_episodes) != self.num_concurrent_episodes:
            logging.warning("You change the num_concurrent_episodes during training!")
            # TODO(pzh) evaluate and training may have different batch size
            #  so we do this as a workaround to make same agent compatible for two usages.
            #  But this is not a final solution. We should remove num_concurrent_episodes from agents finally.
            self.num_concurrent_episodes = len(if_new_episodes)
            self.episode_ids = set(range(self.num_concurrent_episodes))
            self.prev_state = {episode_id: None for episode_id in self.episode_ids}
            self.active_episodes = {episode_id: False for episode_id in self.episode_ids}

        # TODO(pzh) this is a potential risk! If_new_episodes is a list and its order is problematic.
        for ep_id, is_start in enumerate(if_new_episodes):
            self.active_episodes[ep_id] = True  # Set all episodes to activate
            if is_start:
                self.prev_state[ep_id] = None
            elif self.prev_state[ep_id] is not None:
                self.prev_state[ep_id] = [s.detach() for s in self.prev_state[ep_id]]
            else:
                logging.warning("Previous state is None, and the episode is not restart now, why does this happen?")

    def _before_forward(self, end_episode_ids):
        """ Call this function each step before conducting forward pass

        end_episode_ids is equal to step_data["end_index"]
        """
        if end_episode_ids:
            assert max(end_episode_ids) < self.num_concurrent_episodes

        # update agent states
        end_episode_ids = set(end_episode_ids)
        for ep_id in self.episode_ids:
            self.active_episodes[ep_id] = ep_id not in end_episode_ids

        # aggregate previous state
        batch_prev_states = [self.prev_state[ep_id] for ep_id, activate in self.active_episodes.items() if activate]
        return batch_prev_states

    def _after_forward(self, next_states):
        """ Call this function at the end of forward pass to properly deal with previous states """
        activate_episode_index = 0
        for ep_id, activate in self.active_episodes.items():
            if activate:
                # self.prev_state[ep_id] = [element.detach() for element in next_states[activate_episode_index]]
                self.prev_state[ep_id] = next_states[activate_episode_index]
                activate_episode_index += 1
            else:
                self.prev_state[ep_id] = None
        assert activate_episode_index == len(next_states), (activate_episode_index, len(next_states))

    def compute_action(self, step_data, mode, prev_states=None, temperature=None):
        """ Forward pass the agent's model to collect its response in the given timestep.

        This function process only single step data, while all data are consider in a batch form.
        For each new trajectory (a batch of steps data), remember to call
        self.reset_previous_state(data[0]['start_step']) before running into a loop of this function.
        """
        # FIXME(pzh) remove all 'mode' in any part of the model. it's too counter-intuitive
        assert mode in ["mimic", "evaluate"]
        assert PREV_STATE not in step_data
        assert len(step_data["entity_info"]) <= self.num_concurrent_episodes

        # if self.use_cuda:
        #     step_data = to_device(step_data, 'cuda')

        step_data[PREV_STATE] = prev_states or self._before_forward(step_data["end_index"])

        logits = actions = None
        context = NullContext() if self.is_training else torch.no_grad()
        with context:
            if mode == "mimic":
                assert temperature is not None
                logits, next_states = self.model(step_data, mode=mode, temperature=temperature)
            else:
                assert temperature is None
                actions, next_states = self.model(step_data, mode="evaluate")

        self._after_forward(next_states)

        return AgentOutput(actions=actions, logits=logits, next_state=next_states)

    def compute_value(self, step_data):
        # Corresponding to value mode of original ActorCritic
        raise NotImplementedError()

    def get_model(self):
        return self.model

    def eval(self):
        self.model.eval()
        assert not self.is_training

    def train(self):
        # TODO(pzh) this method name is quite important. consider remove this function.
        #  maybe future we introduce agent.train function (which is quiet good honestly)
        self.model.train()
        assert self.is_training

    def __repr__(self):
        return str(self.model)
