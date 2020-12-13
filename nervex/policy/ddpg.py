from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple
import torch

from nervex.torch_utils import Adam
from nervex.rl_utils import v_1step_td_data, v_1step_td_error
from nervex.model import QAC
from nervex.agent import Agent
from .base_policy import Policy, register_policy
from .common_policy import CommonPolicy


class DDPGPolicy(CommonPolicy):

    def _init_learn(self) -> None:
        self._optimizer_actor = Adam(
            self._model.actor.parameters(),
            lr=self._cfg.learn.learning_rate_actor,
            weight_decay=self._cfg.learn.weight_decay
        )
        self._optimizer_critic = Adam(
            self._model.critic.parameters(),
            lr=self._cfg.learn.learning_rate_critic,
            weight_decay=self._cfg.learn.weight_decay
        )
        self._agent = Agent(self._model)
        algo_cfg = self._cfg.learn.algo
        self._algo_cfg_learn = algo_cfg
        self._gamma = algo_cfg.discount_factor
        self._actor_update_freq = algo_cfg.actor_update_freq
        self._use_twin_critic = algo_cfg.use_twin_critic

        self._agent.add_model('target', update_type='momentum', update_kwargs={'theta': algo_cfg.target_theta})
        if algo_cfg.use_noise:
            self._agent.add_plugin(
                'target',
                'action_noise',
                noise_type='gauss',
                noise_kwargs={
                    'mu': 0.0,
                    'sigma': algo_cfg.noise_sigma
                },
                noise_range=algo_cfg.noise_range,
                action_range=algo_cfg.action_range
            )
        self._agent.add_plugin('main', 'grad', enable_grad=True)
        self._agent.add_plugin('target', 'grad', enable_grad=False)
        self._agent.mode(train=True)
        self._agent.target_mode(train=True)
        self._agent.reset()
        self._agent.target_reset()
        self._learn_setting_set = {}
        self._forward_learn_cnt = 0

    def _forward_learn(self, data: dict) -> Dict[str, Any]:
        loss_dict = {}
        # critic learn forward
        next_obs = data.get('next_obs')
        reward = data.get('reward')
        reward = (reward - reward.mean()) / (reward.std() + 1e-8)
        action = data.get('action')

        q_value = self._agent.forward(data, param={'mode': 'compute_q'})['q_value']
        next_data = {'obs': next_obs}
        next_action = self._agent.target_forward(next_data, param={'mode': 'compute_action'})['action']
        next_data['action'] = next_action
        target_q_value = self._agent.target_forward(next_data, param={'mode': 'compute_q'})['q_value']
        if self._use_twin_critic:
            target_q_value = torch.min(target_q_value[0], target_q_value[1])
            td_data = v_1step_td_data(q_value[0], target_q_value, reward, data['done'], data['weight'])
            critic_loss = v_1step_td_error(td_data, self._gamma)
            loss_dict['critic_loss'] = critic_loss
            td_data_twin = v_1step_td_data(q_value[1], target_q_value, reward, data['done'], data['weight'])
            critic_twin_loss = v_1step_td_error(td_data_twin, self._gamma)
            loss_dict['critic_twin_loss'] = critic_twin_loss
        else:
            td_data = v_1step_td_data(q_value, target_q_value, reward, data['done'], data['weight'])
            critic_loss = v_1step_td_error(td_data, self._gamma)
            loss_dict['critic_loss'] = critic_loss
        # critic update
        self._optimizer_critic.zero_grad()
        for k in loss_dict:
            if 'critic' in k:
                loss_dict[k].backward()
        self._optimizer_critic.step()
        # actor learn forward
        if (self._forward_learn_cnt + 1) % self._actor_update_freq == 0:
            actor_loss = -self._agent.forward(data, param={'mode': 'optimize_actor'})['q_value'].mean()
            loss_dict['actor_loss'] = actor_loss
            # actor update
            self._optimizer_actor.zero_grad()
            actor_loss.backward()
            self._optimizer_actor.step()

        # after update
        loss_dict['total_loss'] = sum(loss_dict.values())
        self._forward_learn_cnt += 1
        self._agent.target_update(self._agent.state_dict()['model'])
        return {
            'cur_lr_actor': self._optimizer_actor.defaults['lr'],
            'cur_lr_critic': self._optimizer_critic.defaults['lr'],
            **loss_dict
        }

    def _init_collect(self) -> None:
        self._get_traj_length = self._cfg.collect.get_traj_length
        self._collect_agent = Agent(self._model)
        algo_cfg = self._cfg.collect.algo
        self._collect_agent.add_plugin(
            'main',
            'action_noise',
            noise_type='gauss',
            noise_kwargs={
                'mu': 0.0,
                'sigma': algo_cfg.noise_sigma
            },
            noise_range=None,
            action_range=algo_cfg.action_range
        )
        self._collect_agent.add_plugin('main', 'grad', enable_grad=False)
        self._collect_agent.mode(train=False)
        self._collect_agent.reset()
        self._collect_setting_set = {}

    def _forward_collect(self, data: dict) -> dict:
        output = self._collect_agent.forward(data, param={'mode': 'compute_action'})
        return output

    def _process_transition(self, obs: Any, agent_output: dict, timestep: namedtuple) -> dict:
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'action': agent_output['action'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _init_eval(self) -> None:
        self._eval_agent = Agent(self._model)
        self._eval_agent.add_plugin('main', 'grad', enable_grad=False)
        self._eval_agent.mode(train=False)
        self._eval_agent.reset()
        self._eval_setting_set = {}

    def _forward_eval(self, data: dict) -> dict:
        output = self._eval_agent.forward(data, param={'mode': 'compute_action'})
        return output

    def _init_command(self) -> None:
        pass

    def _create_model_from_cfg(self, cfg: dict) -> torch.nn.Module:
        return QAC(**cfg.model)


register_policy('ddpg', DDPGPolicy)
