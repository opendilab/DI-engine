from copy import deepcopy
import pytest
import torch.nn.functional as F
from typing import Tuple, List, Dict, Any
import torch
from collections import namedtuple
import os

from ding.torch_utils import Adam, to_device
from ding.config import compile_config
from ding.model import model_wrap
from ding.rl_utils import get_train_sample, get_nstep_return_data
from ding.entry import serial_pipeline_il, collect_demo_data, serial_pipeline
from ding.policy import PPOOffPolicy, ILPolicy
from ding.policy.common_utils import default_preprocess_learn
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate
from dizoo.classic_control.cartpole.config import cartpole_dqn_config, cartpole_dqn_create_config, \
    cartpole_ppo_offpolicy_config, cartpole_ppo_offpolicy_create_config


@POLICY_REGISTRY.register('ppo_il')
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

    def _monitor_vars_learn(self) -> list:
        return super()._monitor_vars_learn() + ['policy_loss', 'value_loss']


@pytest.mark.unittest
def test_serial_pipeline_il_ppo():
    # train expert policy
    train_config = [deepcopy(cartpole_ppo_offpolicy_config), deepcopy(cartpole_ppo_offpolicy_create_config)]
    expert_policy = serial_pipeline(train_config, seed=0)

    # collect expert demo data
    collect_count = 10000
    expert_data_path = 'expert_data_ppo.pkl'
    state_dict = expert_policy.collect_mode.state_dict()
    collect_config = [deepcopy(cartpole_ppo_offpolicy_config), deepcopy(cartpole_ppo_offpolicy_create_config)]
    collect_demo_data(
        collect_config, seed=0, state_dict=state_dict, expert_data_path=expert_data_path, collect_count=collect_count
    )

    # il training 1
    il_config = [deepcopy(cartpole_ppo_offpolicy_config), deepcopy(cartpole_ppo_offpolicy_create_config)]
    il_config[0].policy.learn.train_epoch = 20
    il_config[0].policy.type = 'ppo_il'
    _, converge_stop_flag = serial_pipeline_il(il_config, seed=314, data_path=expert_data_path)
    assert converge_stop_flag

    os.popen('rm -rf ' + expert_data_path)


@POLICY_REGISTRY.register('dqn_il')
class DQNILPolicy(ILPolicy):

    def _forward_learn(self, data: dict) -> dict:
        for d in data:
            if isinstance(d['obs'], torch.Tensor):
                d['obs'] = {'processed_obs': d['obs']}
            else:
                assert 'processed_obs' in d['obs']
        return super()._forward_learn(data)

    def _init_collect(self) -> None:
        self._unroll_len = self._cfg.collect.unroll_len
        self._gamma = self._cfg.discount_factor  # necessary for parallel
        self._nstep = self._cfg.nstep  # necessary for parallel
        self._collect_model = model_wrap(self._model, wrapper_name='argmax_sample')
        self._collect_model.reset()

    def _forward_collect(self, data: dict):
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._collect_model.eval()
        with torch.no_grad():
            output = self._collect_model.forward(data)
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
        data = get_nstep_return_data(data, self._nstep, gamma=self._gamma)
        return get_train_sample(data, unroll_len=self._unroll_len)

    def _forward_eval(self, data: dict) -> dict:
        new_data = {id: {'obs': {'processed_obs': t}} for id, t in data.items()}
        return super()._forward_eval(new_data)

    def default_model(self) -> Tuple[str, List[str]]:
        return 'dqn', ['ding.model.template.q_learning']


@pytest.mark.unittest
def test_serial_pipeline_il_dqn():
    # train expert policy
    train_config = [deepcopy(cartpole_dqn_config), deepcopy(cartpole_dqn_create_config)]
    expert_policy = serial_pipeline(train_config, seed=0)

    # collect expert demo data
    collect_count = 10000
    expert_data_path = 'expert_data_dqn.pkl'
    state_dict = expert_policy.collect_mode.state_dict()
    collect_config = [deepcopy(cartpole_dqn_config), deepcopy(cartpole_dqn_create_config)]
    collect_config[0].policy.type = 'dqn_il'
    collect_demo_data(
        collect_config, seed=0, state_dict=state_dict, expert_data_path=expert_data_path, collect_count=collect_count
    )

    # il training 2
    il_config = [deepcopy(cartpole_dqn_config), deepcopy(cartpole_dqn_create_config)]
    il_config[0].policy.learn.train_epoch = 15
    il_config[0].policy.type = 'dqn_il'
    il_config[0].env.stop_value = 50
    _, converge_stop_flag = serial_pipeline_il(il_config, seed=314, data_path=expert_data_path)
    assert converge_stop_flag
    os.popen('rm -rf ' + expert_data_path)
