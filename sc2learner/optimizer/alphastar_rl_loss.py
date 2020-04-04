"""
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. base class for supervised learning on linklink, including basic processes.
"""
import collections
from collections import namedtuple

import torch
import torch.nn.functional as F

from sc2learner.optimizer.base_loss import BaseLoss
from sc2learner.torch_utils import MultioutputsLoss, build_criterion, levenshtein_distance, hamming_distance
from sc2learner.rl_utils import td_lambda_loss, vtrace_loss, upgo_loss, compute_importance_weights, entropy


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
        self.loss_keys = ['total', 'td_lambda', 'vtrace', 'upgo', 'kl', 'action_type_kl', 'entropy']
        self.rollout_outputs = namedtuple(
            "rollout_outputs", [
                'target_outputs', 'behaviour_outputs', 'teacher_outputs', 'baselines', 'rewards', 'actions',
                'game_seconds'
            ]
        )
        self.agent = agent

        self.T = train_config.trajectory_len
        self.vtrace_rhos_min_clip = train_config.vtrace.min_clip
        self.upgo_rhos_min_clip = train_config.upgo.min_clip
        self.action_output_types = train_config.action_output_types
        assert (all([t in ['value', 'logit'] for t in self.action_output_types]))
        self.action_type_kl_seconds = train_config.action_type_kl_seconds
        self.build_order_location_max_limit = train_config.build_order_location_max_limit
        self.use_target_state = train_config.use_target_state

        self.location_expand_ratio = model_config.policy.location_expand_ratio
        self.location_output_type = model_config.policy.head.location_head.output_type

        self.loss_weights = train_config.loss_weights
        self.temperature_scheduler = build_temperature_scheduler(
            train_config.temperature
        )  # get naive temperature scheduler

    def register_log(self, variable_record, tb_logger):
        for k in self.loss_keys:
            variable_record.register_var(k + '_loss')
            tb_logger.register_var(k + '_loss')

    def compute_loss(self, data):
        """
            Overview: Overwrite function, main process of training
            Arguments:
                - data (:obj:`batch_data`): batch_data created by dataloader

        """
        target_outputs, behaviour_outputs, teacher_outputs, baselines, rewards, actions, game_seconds = self._rollout(
            data
        )

        # td_lambda and v_trace
        actor_critic_loss = 0.
        for field, baseline, reward in zip(baselines._fields, baselines, rewards):
            actor_critic_loss += self._td_lambda_loss(baseline, reward) * self.loss_weights.baseline[field]
            actor_critic_loss += self._vtrace_pg_loss(baseline, reward, target_outputs, behaviour_outputs,
                                                      actions) * self.loss_weights.pg[field]
        # upgo loss
        upgo_loss = self._upgo_loss(
            baselines['winloss'], rewards['winloss'], target_outputs, behaviour_outputs, actions
        ) * self.loss_weights.upgo['winloss']

        # human kl loss
        kl_loss, action_type_kl_loss = self._human_kl_loss(target_outputs, teacher_outputs, game_seconds)
        kl_loss *= self.loss_weights.kl
        action_type_kl_loss *= self.loss_weights.action_type_kl
        # entropy loss
        ent_loss = self._entropy_loss(target_outputs) * self.loss_weights.entropy

        total_loss = actor_critic_loss + kl_loss + action_type_kl_loss + ent_loss  # + upgo_loss
        return {
            'total_loss': total_loss,
            'actor_critic_loss': actor_critic_loss,
            'kl_loss': kl_loss,
            'action_type_kl_loss': action_type_kl_loss,
            'ent_loss': ent_loss,
            'upgo_loss': upgo_loss,
        }

    def _rollout(self, data):
        temperature = self.temperature_scheduler.step()
        next_state_home, next_state_away = None, None
        outputs = self.rollout_outputs(*[[] for _ in range(len(self.rollout_outputs._fields))])
        for idx, step_data in enumerate(data):
            if self.use_target_state and next_state_home is not None:
                step_data['home']['prev_state'] = next_state_home
                step_data['away']['prev_state'] = next_state_away
            target_outputs, baselines, next_state_home, next_state_away = self.agent.compute_action_value(
                step_data, temperature
            )
            # add to outputs
            home = step_data['home']
            outputs.target_outputs.append(target_outputs)
            outputs.behaviour_outputs.append(home['behaviour_outputs'])
            outputs.teacher_outputs.append(home['teacher_outputs'])
            outputs.baselines.append(baselines)
            outputs.rewards.append(
                self._compute_pseudo_rewards(home['agent_z'], home['target_z'], home['rewards'], home['game_seconds'])
            )
            outputs.actions.append(home['actions'])
        # last baselines/values
        last_obs = {'home': data[-1]['home_next'], 'away': data[-1]['away_next']}
        if self.use_target_state and next_state_home is not None:
            last_obs['home']['prev_state'] = next_state_home
            last_obs['away']['prev_state'] = next_state_away
        last_baselines = self.agent.compute_action_value(last_obs, temperature)
        outputs = list(zip(*outputs))
        # add game_seconds
        outputs.append(data[0]['home']['game_seconds'])
        return outputs

    def _compute_pseudo_rewards(self, agent_z, target_z, rewards, game_seconds):
        """
            Overview: compute pseudo rewards from human replay z
            Arguments:
                - agent_z (:obj:`dict`)
                - target_z (:obj:`dict`)
                - rewards (:obj:`torch.Tensor`)
                - game_seconds (:obj:`int`)
            Returns:
                - rewards (:obj:`dict`): a dict contains different type rewards
        """
        def loc_fn(p1, p2, max_limit=self.build_order_location_max_limit):
            dist = F.l1_loss(p1, p2, reduction='sum')
            dist = dist.clamp(0, max_limit)
            return dist

        def get_time_factor():
            if game_seconds < 8 * 60:
                return 1.0
            elif game_seconds < 16 * 60:
                return 0.5
            elif game_seconds < 24 * 60:
                return 0.25
            else:
                return 0

        new_rewards = {}
        new_rewards['winloss'] = rewards
        new_rewards['build_order'] = levenshtein_distance(
            agent_z['build_order']['type'], target_z['build_order']['type'], agent_z['build_order']['loc'],
            target_z['build_order']['loc'], loc_fn
        )
        for k in ['built_units', 'upgrades', 'effects']:
            new_rewards[k] = hamming_distance(agent_z[k], target_z[k])
        factor = get_time_factor()
        new_rewards = {k: v * factor for k, v in new_rewards.items()}
        return new_rewards

    def _td_lambda_loss(self, baseline, reward):
        """
            default: gamma=1.0, lamda=0.8
        """
        assert (isinstance(baseline, torch.Tensor) and baseline.shape[0] == self.T + 1)
        assert (isinstance(reward, torch.Tensor) and reward.shape[0] == self.T)
        return td_lambda_loss(baseline, reward)

    def _vtrace_pg_loss(self, baseline, reward, target_outputs, behaviour_outputs, actions):
        """
            seperated vtrace loss
        """
        def _vtrace(target_output, behaviour_output, action, action_output_type):
            clipped_rhos = compute_importance_weights(
                target_output, behaviour_output, action_output_type, action, min_clip=self.vtrace_rhos_min_clip
            )
            clipped_cs = clipped_rhos
            return vtrace_loss(target_output, action_output_type, clipped_rhos, clipped_cs, action, reward, baseline)

        loss = 0.
        for k in self.action_keys:
            loss += _vtrace(target_outputs[k], behaviour_outputs[k], actions[k], self.action_output_types)

        return loss

    def _upgo_loss(self, baseline, reward, target_outputs, behaviour_outputs, actions):
        def _upgo(target_output, behaviour_output, action, action_output_type):
            clipped_rhos = compute_importance_weights(
                target_output, behaviour_output, action_output_type, action, min_clip=self.upgo_rhos_min_clip
            )
            return upgo_loss(target_output, action_output_type, clipped_rhos, action, reward, baseline)

        loss = 0.
        for k in self.action_keys:
            loss += _upgo(target_outputs[k], behaviour_outputs[k], actions[k], self.action_output_types)
        return loss

    def _human_kl_loss(self, target_outputs, teacher_outputs, game_seconds):
        kl_loss = 0.
        for k in self.action_keys:
            if self.action_output_types[k] == 'logit':
                target_output = F.log_softmax(target_outputs[k], dim=2)
                teacher_output = F.softmax(teacher_outputs[k], dim=2)
                kl_loss += F.kl_div(target_output, teacher_output)
            elif self.action_output_types[k] == 'value':
                target_output, teacher_output = target_outputs[k], teacher_outputs[k]
                kl_loss += F.l1_loss(target_output, teacher_output)

        if game_seconds < self.action_type_kl_seconds:
            target_output = F.log_softmax(target_outputs['action_type'], dim=2)
            teacher_output = F.softmax(teacher_outputs['action_type'], dim=2)
            action_type_kl_loss = F.kl_div(target_output, teacher_output)
        return kl_loss, action_type_kl_loss

    def _entropy_loss(self, target_outputs):
        loss = 0.
        for k in self.action_keys:
            if self.action_output_types[k] == 'logit':
                loss += entropy(target_outputs[k])
        return loss
