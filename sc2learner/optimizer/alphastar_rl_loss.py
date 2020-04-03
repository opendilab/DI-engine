"""
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. base class for supervised learning on linklink, including basic processes.
"""
import collections

import torch
import torch.nn.functional as F

from sc2learner.optimizer.base_loss import BaseLoss
from sc2learner.torch_utils import MultiLogitsLoss, build_criterion
from sc2learner.rl_utils import td_lambda_loss, vtrace_loss, compute_importance_weights, entropy


def build_temperature_scheduler(temperature):
    """
        Overview: use config to initialize scheduler. Use naive temperature scheduler as default.
        Arguments:
            - cfg (:obj:`dict`): scheduler config
        Returns:
            - (:obj`Scheduler`): scheduler created by this function
    """
    class ConstantTemperatureSchedule:
        def __init__(self, init_val=0.1):
            self.init_val = init_val

        def step(self):
            return self.init_val

        def value(self):
            return self.init_val

    return ConstantTemperatureSchedule(init_val=temperature)


class AlphaStarSupervisedLoss(BaseLoss):
    def __init__(self, agent, train_config, model_config):
        self.action_keys = ['action_type', 'delay', 'queued', 'selected_units', ' target_units', 'target_location']
        self.agent = agent

        self.T = train_config.trajectory_len
        self.vtrace_rhos_min_clip = train_config.vtrace.min_clip
        self.entropy_keys = train_config.entropy.keys

        self.location_expand_ratio = model_config.policy.location_expand_ratio
        self.location_output_type = model_config.policy.head.location_head.output_type

        self.loss_weights = train_config.loss_weights
        self.temperature_scheduler = build_temperature_scheduler(
            train_config.temperature
        )  # get naive temperature scheduler

    def register_log(self, variable_record, tb_logger):
        for k in self.loss_func.keys():
            variable_record.register_var(k + '_loss')
            tb_logger.register_var(k + '_loss')

    def compute_loss(self, data):
        """
            Overview: Overwrite function, main process of training
            Arguments:
                - data (:obj:`batch_data`): batch_data created by dataloader

        """
        target_logits, behaviour_logits, teacher_logits, baselines, rewards, actions, game_seconds = self._rollout(data)

        # td_lambda and v_trace
        actor_critic_loss = 0.
        for field, baseline, reward in zip(baselines._fields, baselines, rewards):
            actor_critic_loss += self._td_lambda_loss(baseline, reward) * self.loss_weights.baseline[field]
            actor_critic_loss += self._vtrace_pg_loss(baseline, reward, target_logits,
                                                      behaviour_logits) * self.loss_weights.pg[field]
        # upgo loss

        # human kl loss
        kl_loss, action_type_kl_loss = self._human_kl_loss(target_logits, teacher_logits, game_seconds)
        kl_loss *= self.loss_weights.kl
        action_type_kl_loss *= self.loss_weights.action_type_kl
        # entropy loss
        ent_loss = self._entropy_loss(target_logits) * self.loss_weights.entropy

        total_loss = actor_critic_loss + kl_loss + action_type_kl_loss + ent_loss  # + upgo_loss
        return {
            'total_loss': total_loss,
            'actor_critic_loss': actor_critic_loss,
            'kl_loss': kl_loss,
            'action_type_kl_loss': action_type_kl_loss,
            'ent_loss': ent_loss,
            #'upgo_loss': upgo_loss,
        }

    def _rollout(self, data):
        temperature = self.temperature_scheduler.step()
        for idx, step_data in enumerate(data):
            rewards = self._compute_pseudo_rewards(step_data)
            target_logits, baselines, next_state = self.agent.compute_action(step_data, 'step', temperature=temperature)

    def _compute_pseudo_rewards(self):
        """

        Returns:
            - rewards (:obj:`dict`): a dict contains different type rewards
        Note:
            agent_z, target_z, reward
        """

    def _td_lambda_loss(self, baseline, reward):
        """
            default: gamma=1.0, lamda=0.8
        """
        assert (isinstance(baseline, torch.Tensor) and baseline.shape[0] == self.T + 1)
        assert (isinstance(reward, torch.Tensor) and reward.shape[0] == self.T)
        return td_lambda_loss(baseline, reward)

    def _vtrace_pg_loss(self, baseline, reward, target_logits, behaviour_logits, actions):
        """
            seperated vtrace loss
        """
        def _vtrace(target_logit, behaviour_logit, action):
            clipped_rhos = compute_importance_weights(
                target_logit, behaviour_logit, action, min_clip=self.vtrace_rhos_min_clip
            )
            clipped_cs = clipped_rhos
            return vtrace_loss(target_logit, clipped_rhos, clipped_cs, action, reward, baseline)

        loss = 0.
        for k in self.action_keys:
            loss += _vtrace(target_logits[k], behaviour_logits[k], actions[k])

        return loss

    def _upgo_loss(self):
        pass

    def _human_kl_loss(self, target_logits, teacher_logits, masks, game_seconds):
        pass

    def _entropy_loss(self, target_logits):
        loss = 0.
        for k in self.entropy_keys:
            loss += entropy(target_logits[k])
        return loss
