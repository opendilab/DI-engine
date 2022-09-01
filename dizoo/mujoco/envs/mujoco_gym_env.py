import numpy as np

import gym
from gym.envs.mujoco.ant import AntEnv
from gym.envs.mujoco.humanoid import HumanoidEnv


def gym_env_register(id, max_episode_steps=1000):

    def register(gym_env):
        spec = {
            'id': id,
            'entry_point': (f'dizoo.mujoco.envs.mujoco_gym_env:{gym_env.__name__}'),
            'max_episode_steps': max_episode_steps
        }
        gym.register(**spec)
        return gym_env

    return register


@gym_env_register('AntTruncatedObs-v2')
class AntTruncatedObsEnv(AntEnv):
    """
    Overview:
        Modified ant with observation dim truncated to 27, which is used in MBPO (arXiv: 1906.08253).
    .. note::
        External forces (sim.data.cfrc_ext) are removed from the observation.
        Otherwise identical to Ant-v2 from\
        <https://github.com/openai/gym/blob/master/gym/envs/mujoco/ant.py>.
    """

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
                # np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]
        )


@gym_env_register('HumanoidTruncatedObs-v2')
class HumanoidTruncatedObsEnv(HumanoidEnv):
    """
    Overview:
        Modified humanoid with observation dim truncated to 45, which is used in MBPO (arXiv: 1906.08253). 
    .. note::
        COM inertia (cinert), COM velocity (cvel), actuator forces (qfrc_actuator),\
        and external forces (cfrc_ext) are removed from the observation.
        Otherwise identical to Humanoid-v2 from\
        <https://github.com/openai/gym/blob/master/gym/envs/mujoco/humanoid.py>.
    """

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate(
            [
                data.qpos.flat[2:],
                data.qvel.flat,
                # data.cinert.flat,
                # data.cvel.flat,
                # data.qfrc_actuator.flat,
                # data.cfrc_ext.flat
            ]
        )
