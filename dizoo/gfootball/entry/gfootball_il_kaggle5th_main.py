from dizoo.gfootball.entry.gfootball_il_config import gfootball_il_main_config, gfootball_il_create_config
from dizoo.gfootball.model.bots.kaggle_5th_place_model import FootballKaggle5thPlaceModel
from dizoo.gfootball.model.iql.iql_network import FootballIQL

from ding.config import read_config, compile_config
from ding.policy import create_policy
from copy import deepcopy
from typing import Tuple, List, Dict, Any
import torch
from collections import namedtuple
import os
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)

from ding.torch_utils import Adam, to_device
from ding.rl_utils import get_train_sample, get_nstep_return_data
from ding.entry import serial_pipeline_bc, collect_demo_data, serial_pipeline
from ding.policy import PPOOffPolicy, DiscreteBehaviourCloningPolicy
from ding.utils import POLICY_REGISTRY
from ding.utils.data import default_collate, default_decollate


@POLICY_REGISTRY.register('iql_bc')
class IQLILPolicy(DiscreteBehaviourCloningPolicy):

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
            # o = default_decollate(self._eval_model.forward(data))
            if self._cuda:
                data = to_device(data, self._device)
            self._eval_model.eval()
            with torch.no_grad():
                output = self._eval_model.forward(data)
            if self._cuda:
                output = to_device(output, 'cpu')
            output = default_decollate(output)
            return {i: d for i, d in zip(data_id, output)}
        return self._model(data)

    def default_model(self) -> Tuple[str, List[str]]:
        return 'football_iql', ['dizoo.gfootball.model.iql']


# demo_transitions = 2  # debug
demo_transitions = int(3e5)  # key hyper-parameter
data_path_transitions = dir_path + f'/gfootball_kaggle5th_{demo_transitions}-demo-transitions.pkl'
gfootball_il_main_config.exp_name = 'data_gfootball/gfootball_il_kaggle5th_seed0'
seed=0

"""
phase 1: train/obtain expert policy
"""
train_config = [deepcopy(gfootball_il_main_config), deepcopy(gfootball_il_create_config)]
input_cfg = train_config
if isinstance(input_cfg, str):
    cfg, create_cfg = read_config(input_cfg)
else:
    cfg, create_cfg = input_cfg
create_cfg.policy.type = create_cfg.policy.type + '_command'
env_fn = None
cfg = compile_config(cfg, seed=seed, env=env_fn, auto=True, create_cfg=create_cfg, save_cfg=True)

football_kaggle_5th_place_model = FootballKaggle5thPlaceModel()
expert_policy = create_policy(cfg.policy, model=football_kaggle_5th_place_model,
                              enable_field=['learn', 'collect', 'eval', 'command'])

# collect expert demo data
state_dict = expert_policy.collect_mode.state_dict()
collect_config = [deepcopy(gfootball_il_main_config), deepcopy(gfootball_il_create_config)]
collect_demo_data(
    collect_config, seed=seed, data_path_transitions=data_path_transitions, collect_count=demo_transitions,
    model=football_kaggle_5th_place_model, state_dict=state_dict,
)

"""
phase 2: il training
"""
il_config = [deepcopy(gfootball_il_main_config), deepcopy(gfootball_il_create_config)]
il_config[0].policy.learn.train_epoch = 50  # key hyper-parameter
# il_config[0].policy.learn.train_epoch = 2  # debug
il_config[0].policy.type = 'iql_bc'
il_config[0].env.stop_value = 999  # Don't stop until training <train_epoch> epochs
il_config[0].policy.eval.evaluator.multi_gpu = False
football_iql_model = FootballIQL()
_, converge_stop_flag = serial_pipeline_bc(il_config, seed=seed, data_path=data_path_transitions, model=football_iql_model)
# assert converge_stop_flag
# os.popen('rm -rf ' + data_path_transitions)
