from copy import deepcopy
import pytest
import torch.nn.functional as F
from typing import Tuple, List, Dict, Any
import torch
from collections import namedtuple
import os

from ding.torch_utils import to_device
from ding.rl_utils import get_train_sample, get_nstep_return_data
from ding.entry import serial_pipeline_bc, collect_demo_data, serial_pipeline
from ding.policy import PPOOffPolicy, BehaviourCloningPolicy
from ding.policy.common_utils import default_preprocess_learn
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate
from dizoo.classic_control.cartpole.config import cartpole_dqn_config, cartpole_dqn_create_config, \
    cartpole_offppo_config, cartpole_offppo_create_config
from dizoo.classic_control.pendulum.config import pendulum_sac_config, pendulum_sac_create_config


@POLICY_REGISTRY.register('ppo_bc')
class PPOILPolicy(PPOOffPolicy):

    def _forward_learn(self, data: dict) -> dict:
        data = default_preprocess_learn(data, ignore_done=self._cfg.learn.get('ignore_done', False), use_nstep=False)
        self._learn_model.train()
        output = self._learn_model.forward(data['obs'], mode='compute_actor_critic')
        value_loss = F.mse_loss(output['value'], data['value'])
        policy_loss = F.smooth_l1_loss(output['logit'], data['logit'])
        total_loss = value_loss + policy_loss
        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()
        return {
            'cur_lr': self._optimizer.defaults['lr'],
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
        }

    def _forward_eval(self, data):
        if isinstance(data, dict):
            data_id = list(data.keys())
            data = default_collate(list(data.values()))
            o = default_decollate(self._eval_model.forward(data, mode='compute_actor'))
            return {i: d for i, d in zip(data_id, o)}
        return self._model(data, mode='compute_actor')

    def _monitor_vars_learn(self) -> list:
        return super()._monitor_vars_learn() + ['policy_loss', 'value_loss']


@pytest.mark.unittest
def test_serial_pipeline_bc_ppo():
    # train expert policy
    train_config = [deepcopy(cartpole_offppo_config), deepcopy(cartpole_offppo_create_config)]
    train_config[0].exp_name = 'test_serial_pipeline_bc_ppo'
    expert_policy = serial_pipeline(train_config, seed=0)

    # collect expert demo data
    collect_count = 10000
    expert_data_path = 'expert_data_ppo_bc.pkl'
    state_dict = expert_policy.collect_mode.state_dict()
    collect_config = [deepcopy(cartpole_offppo_config), deepcopy(cartpole_offppo_create_config)]
    collect_config[0].exp_name = 'test_serial_pipeline_bc_ppo_collect'
    collect_demo_data(
        collect_config, seed=0, state_dict=state_dict, expert_data_path=expert_data_path, collect_count=collect_count
    )

    # il training 1
    il_config = [deepcopy(cartpole_offppo_config), deepcopy(cartpole_offppo_create_config)]
    il_config[0].policy.eval.evaluator.multi_gpu = False
    il_config[0].policy.learn.train_epoch = 20
    il_config[1].policy.type = 'ppo_bc'
    il_config[0].policy.continuous = False
    il_config[0].exp_name = 'test_serial_pipeline_bc_ppo_il'
    _, converge_stop_flag = serial_pipeline_bc(il_config, seed=314, data_path=expert_data_path)
    assert converge_stop_flag

    os.popen('rm -rf ' + expert_data_path)


@POLICY_REGISTRY.register('dqn_bc')
class DQNILPolicy(BehaviourCloningPolicy):

    def _forward_learn(self, data: dict) -> dict:
        return super()._forward_learn(data)

    def _forward_collect(self, data: dict, eps: float):
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._collect_model.eval()
        with torch.no_grad():
            output = self._collect_model.forward(data, eps=eps)
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _process_transition(self, obs: Any, model_output: dict, timestep: namedtuple) -> Dict[str, Any]:
        ret = super()._process_transition(obs, model_output, timestep)
        ret['next_obs'] = timestep.obs
        return ret

    def _get_train_sample(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        super()._get_train_sample(data)
        data = get_nstep_return_data(data, 1, gamma=0.99)
        return get_train_sample(data, unroll_len=1)

    def _forward_eval(self, data: dict) -> dict:
        if isinstance(data, dict):
            data_id = list(data.keys())
            data = default_collate(list(data.values()))
            o = default_decollate(self._eval_model.forward(data))
            return {i: d for i, d in zip(data_id, o)}
        return self._model(data)

    def default_model(self) -> Tuple[str, List[str]]:
        return 'dqn', ['ding.model.template.q_learning']


@pytest.mark.unittest
def test_serial_pipeline_bc_dqn():
    # train expert policy
    train_config = [deepcopy(cartpole_dqn_config), deepcopy(cartpole_dqn_create_config)]
    expert_policy = serial_pipeline(train_config, seed=0)

    # collect expert demo data
    collect_count = 10000
    expert_data_path = 'expert_data_dqn.pkl'
    state_dict = expert_policy.collect_mode.state_dict()
    collect_config = [deepcopy(cartpole_dqn_config), deepcopy(cartpole_dqn_create_config)]
    collect_config[1].policy.type = 'dqn_bc'
    collect_config[0].policy.continuous = False
    collect_config[0].policy.other.eps = 0
    collect_demo_data(
        collect_config, seed=0, state_dict=state_dict, expert_data_path=expert_data_path, collect_count=collect_count
    )

    # il training 2
    il_config = [deepcopy(cartpole_dqn_config), deepcopy(cartpole_dqn_create_config)]
    il_config[0].policy.learn.train_epoch = 15
    il_config[1].policy.type = 'dqn_bc'
    il_config[0].policy.continuous = False
    il_config[0].env.stop_value = 50
    il_config[0].policy.eval.evaluator.multi_gpu = False
    _, converge_stop_flag = serial_pipeline_bc(il_config, seed=314, data_path=expert_data_path)
    assert converge_stop_flag
    os.popen('rm -rf ' + expert_data_path)


@pytest.mark.unittest
def test_serial_pipeline_bc_sac():
    # train expert policy
    train_config = [deepcopy(pendulum_sac_config), deepcopy(pendulum_sac_create_config)]
    expert_policy = serial_pipeline(train_config, seed=0, max_train_iter=10)

    # collect expert demo data
    collect_count = 10000
    expert_data_path = 'expert_data_sac.pkl'
    state_dict = expert_policy.collect_mode.state_dict()
    collect_config = [deepcopy(pendulum_sac_config), deepcopy(pendulum_sac_create_config)]
    collect_demo_data(
        collect_config, seed=0, state_dict=state_dict, expert_data_path=expert_data_path, collect_count=collect_count
    )

    # il training 2
    il_config = [deepcopy(pendulum_sac_config), deepcopy(pendulum_sac_create_config)]
    il_config[0].policy.learn.train_epoch = 15
    il_config[1].policy.type = 'bc'
    il_config[0].policy.continuous = True
    il_config[0].env.stop_value = 50
    il_config[0].policy.model = dict(
        obs_shape=3,
        action_shape=1,
        action_space='regression',
        actor_head_hidden_size=128,
    )
    il_config[0].policy.loss_type = 'l1_loss'
    il_config[0].policy.learn.learning_rate = 1e-5
    il_config[0].policy.eval.evaluator.multi_gpu = False
    il_config[1].policy.type = 'bc'
    _, converge_stop_flag = serial_pipeline_bc(il_config, seed=314, data_path=expert_data_path, max_iter=10)
    os.popen('rm -rf ' + expert_data_path)
