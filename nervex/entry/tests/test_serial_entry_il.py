from copy import deepcopy

import pytest
import torch.nn.functional as F

from app_zoo.classic_control.cartpole.config import cartpole_ppo_config, cartpole_ppo_create_config
from nervex.entry import serial_pipeline_il, collect_demo_data, serial_pipeline
from nervex.policy import PPOPolicy
from nervex.policy.common_utils import default_preprocess_learn
from nervex.utils import POLICY_REGISTRY


@POLICY_REGISTRY.register('ppo_il')
class PPOILPolicy(PPOPolicy):

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
def test_serial_pipeline_il():
    # train expert policy
    train_config = [deepcopy(cartpole_ppo_config), deepcopy(cartpole_ppo_create_config)]
    expert_policy = serial_pipeline(train_config, seed=0)

    # collect expert demo data
    collect_count = 10000
    expert_data_path = 'expert_data.pkl'
    state_dict = expert_policy.collect_mode.state_dict()
    collect_config = [deepcopy(cartpole_ppo_config), deepcopy(cartpole_ppo_create_config)]
    collect_demo_data(
        collect_config, seed=0, state_dict=state_dict, expert_data_path=expert_data_path, collect_count=collect_count
    )

    # il training
    il_config = [deepcopy(cartpole_ppo_config), deepcopy(cartpole_ppo_create_config)]
    il_config[0].policy.learn.train_epoch = 10
    il_config[0].policy.type = 'ppo_il'
    _, converge_stop_flag = serial_pipeline_il(il_config, seed=314, data_path=expert_data_path)
    assert converge_stop_flag
