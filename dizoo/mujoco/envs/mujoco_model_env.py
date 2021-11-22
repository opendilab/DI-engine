from typing import Any, Union, List, Callable, Dict
import copy
import torch
import torch.nn as nn
import numpy as np

from ding.envs import BaseEnv, BaseEnvTimestep, BaseEnvInfo, update_shape
from ding.envs.common.env_element import EnvElement, EnvElementInfo
from ding.envs.common.common_function import affine_transform
from ding.torch_utils import to_tensor, to_ndarray, to_list
from .mujoco_wrappers import wrap_mujoco
from ding.utils import ENV_REGISTRY
from ding.worker.collector.base_serial_collector import to_tensor_transitions


@ENV_REGISTRY.register('mujoco_model')
class MujocoModelEnv(object):

    def __init__(self, env_id: str, set_rollout_length: Callable, rollout_batch_size: int = 100000):
        self.env_id = env_id
        self.rollout_batch_size = rollout_batch_size
        self._set_rollout_length = set_rollout_length

    def termination_fn(self, next_obs: torch.Tensor) -> torch.Tensor:
        # This function determines whether each state is a terminated state
        assert len(next_obs.shape) == 2
        if self.env_id == "Hopper-v2":
            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = torch.isfinite(next_obs).all(-1) \
                       * (torch.abs(next_obs[:, 1:]) < 100).all(-1) \
                       * (height > .7) \
                       * (torch.abs(angle) < .2)

            done = ~not_done
            return done
        elif self.env_id == "Walker2d-v2":
            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = (height > 0.8) \
                       * (height < 2.0) \
                       * (angle > -1.0) \
                       * (angle < 1.0)
            done = ~not_done
            return done
        elif 'walker_' in self.env_id:
            torso_height = next_obs[:, -2]
            torso_ang = next_obs[:, -1]
            if 'walker_7' in self.env_id or 'walker_5' in self.env_id:
                offset = 0.
            else:
                offset = 0.26
            not_done = (torso_height > 0.8 - offset) \
                       * (torso_height < 2.0 - offset) \
                       * (torso_ang > -1.0) \
                       * (torso_ang < 1.0)
            done = ~not_done
            return done
        elif self.env_id == "HalfCheetah-v3":
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
