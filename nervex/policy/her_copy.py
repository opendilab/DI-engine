


from typing import List, Dict, Any, Tuple, Union, Optional
from collections import namedtuple, deque
import torch
from easydict import EasyDict
import numpy as np
from .base_policy import Policy, register_policy
from .common_policy import CommonPolicy

from .dqn import DQNPolicy
from .rainbow_dqn import RainbowDQNPolicy
from .ddpg import DDPGPolicy
from .a2c import A2CPolicy
from .ppo import PPOPolicy
from .sac import SACPolicy
from .impala import IMPALAPolicy

policy_dict = {'dqn' : DQNPolicy, 'rainbow_dqn' : RainbowDQNPolicy, 'ddpg':DDPGPolicy}

class HERPolicy(DQNPolicy):
    r"""
    Overview:
        Policy class of HER algorithm.
    """
    def __init__(self,cfg: dict, model_type: Optional[type] = None, enable_field: Optional[List[str]] = None):
        super(HERPolicy, self).__init__(cfg, model_type, enable_field)
        self._replay_strategy =  self._cfg.collect.algo.replay_strategy
        self._state_size =  self._cfg.model.state_size
        self._goal_size =  self._cfg.model.goal_size
        print(self._cfg.base_policy_type)
        self._base_policy_type = self._cfg.base_policy_type
        self._base_policy = policy_dict[self._base_policy_type](cfg, model_type, enable_field)
        if self._replay_strategy != 'final':
            self._replay_k = self._cfg.collect.algo.replay_k
        else:
            self._replay_k = 1

        for attr in ['_create_model_from_cfg', '_data_postprocess_collect', '_data_preprocess_collect',
                 '_data_preprocess_learn', '_forward_collect', '_forward_eval', '_forward_learn',
                 '_get_setting_collect', '_get_setting_eval', '_get_setting_learn', '_init_collect',
                 '_init_command', '_init_eval', '_init_learn', '_monitor_vars_learn', '_reset_collect',
                 '_reset_eval', '_reset_learn', 'collect_function', 'collect_mode', 'command_function',
                 'command_mode', 'eval_function', 'eval_mode', 'learn_function', 'learn_mode',
                 'set_setting', 'state_dict_handle', 'sync_gradients']:
            self.__setattr__(attr, self._base_policy.__getattribute__(attr))

    def _get_train_sample(self, traj_cache: deque) -> Union[None, List[Any]]:
        r"""
        Overview:
            Get the trajectory and the n step return data, then sample from the n_step return data
        Arguments:
            - traj_cache (:obj:`deque`): The trajectory's cache
        Returns:
            - samples (:obj:`dict`): The training samples generated
        """
        # adder is defined in _init_collect
        return_num = 0 if self._collect_nstep == 1 else self._collect_nstep
        assert self._collect_nstep == 1
        # import ipdb;ipdb.set_trace()
        timesteps = []
        for idx in range(len(traj_cache)):
            timestep = {}

            state = traj_cache[idx]['obs'][:self._state_size - self._goal_size]
            next_state = traj_cache[idx]['next_obs'][:self._state_size - self._goal_size]
            for i in range(self._replay_k):

                if self._replay_strategy == 'final':
                    p_idx = -1
                elif self._replay_strategy == 'episode':
                    p_idx = np.random.randint(len(traj_cache))
                elif self._replay_strategy == 'future':
                    p_idx = np.random.randint(idx,len(traj_cache))
                # elif self._replay_strategy == 'random':

                new_goal = traj_cache[-1]['next_obs'][:self._state_size - self._goal_size]

                timestep['obs'] = torch.cat([state,new_goal],dim=0)
                timestep['next_obs'] = torch.cat([next_state,new_goal],dim=0)
                timestep['goal'] = new_goal
                timestep['reward'] = torch.FloatTensor([1]) if torch.all(next_state == new_goal) else torch.FloatTensor([0])
                timestep['action'] = traj_cache[idx]['action']
                timestep['done'] = traj_cache[idx]['done']

            timesteps.append(timestep)

        return timesteps

    def _process_transition(self, obs: Any, agent_output: dict, timestep: namedtuple) -> dict:
        r"""
       Overview:
           Generate dict type transition data from inputs.
       Arguments:
           - obs (:obj:`Any`): Env observation
           - agent_output (:obj:`dict`): Output of collect agent, including at least ['action']
           - timestep (:obj:`namedtuple`): Output after env step, including at least ['obs', 'reward', 'done'] \
               (here 'obs' indicates obs after env step).
       Returns:
           - transition (:obj:`dict`): Dict type transition data.
       """
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'goal':timestep.goal,
            'action': agent_output['action'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return EasyDict(transition)



register_policy('her', HERPolicy)
