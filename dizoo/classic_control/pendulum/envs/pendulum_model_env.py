from typing import Any, Union, List, Callable, Dict
import copy
import torch
import torch.nn as nn
import numpy as np

from ding.envs import BaseEnv, BaseEnvTimestep, BaseEnvInfo, update_shape
from ding.envs.common.env_element import EnvElement, EnvElementInfo
from ding.envs.common.common_function import affine_transform
from ding.torch_utils import to_tensor, to_ndarray, to_list
from ding.utils import ENV_REGISTRY
from ding.worker.collector.base_serial_collector import to_tensor_transitions


@ENV_REGISTRY.register('pendulum_model')
class PendulumModelEnv(object):

    def __init__(self, env_id: str, set_rollout_length: Callable, rollout_batch_size: int = 100000):
        self.env_id = env_id
        self.rollout_batch_size = rollout_batch_size
        self._set_rollout_length = set_rollout_length

    def termination_fn(self, next_obs: torch.Tensor) -> torch.Tensor:
        # This function determines whether each state is a terminated state
        done = torch.zeros_like(next_obs.sum(-1)).bool()
        return done

    def rollout(
            self,
            env_model: nn.Module,
            policy: 'Policy',  # noqa
            replay_buffer: 'IBuffer',  # noqa
            imagine_buffer: 'IBuffer',  # noqa
            envstep: int,
            cur_learner_iter: int
    ) -> None:
        """
        Overview:
            This function samples from the replay_buffer, rollouts to generate new data,
            and push them into the imagine_buffer
        """
        # set rollout length
        rollout_length = self._set_rollout_length(envstep)
        # load data
        data = replay_buffer.sample(self.rollout_batch_size, cur_learner_iter, replace=True)
        obs = {id: data[id]['obs'] for id in range(len(data))}
        # rollout
        buffer = [[] for id in range(len(obs))]
        new_data = []
        for i in range(rollout_length):
            # get action
            obs = to_tensor(obs, dtype=torch.float32)
            policy_output = policy.forward(obs)
            actions = {id: output['action'] for id, output in policy_output.items()}
            # predict next obs and reward
            timesteps = self.step(obs, actions, env_model)
            obs_new = {}
            for id, timestep in timesteps.items():
                transition = policy.process_transition(obs[id], policy_output[id], timestep)
                transition['collect_iter'] = cur_learner_iter
                buffer[id].append(transition)
                if not timestep.done:
                    obs_new[id] = timestep.obs
                if timestep.done or i + 1 == rollout_length:
                    transitions = to_tensor_transitions(buffer[id])
                    train_sample = policy.get_train_sample(transitions)
                    new_data.extend(train_sample)
            if len(obs_new) == 0:
                break
            obs = obs_new

        imagine_buffer.push(new_data, cur_collector_envstep=envstep)

    def step(self, obs: Dict, act: Dict, env_model: nn.Module) -> Dict:
        # This function has the same input and output format as env manager's step
        data_id = list(obs.keys())
        obs = torch.stack([obs[id] for id in data_id], dim=0)
        act = torch.stack([act[id] for id in data_id], dim=0)
        rewards, next_obs = env_model.predict(obs, act)
        terminals = self.termination_fn(next_obs)
        timesteps = {
            id: BaseEnvTimestep(n, r, d, {})
            for id, n, r, d in zip(data_id, next_obs.numpy(), rewards.numpy(), terminals.numpy())
        }
        return timesteps
