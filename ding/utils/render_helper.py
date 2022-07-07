from typing import TYPE_CHECKING, Optional
from numpy import ndarray

if TYPE_CHECKING:
    from ding.envs import BaseEnv, BaseEnvManager


def render(env: "BaseEnv", render_mode: Optional[str] = 'rgb_array') -> "ndarray":
    '''
    Overview:
        Render the environment's current frame.
    Arguments:
        - env (:obj:`BaseEnv`): DI-engine env instance.
        - render_mode (:obj:`str`): Render mode.
    Returns:
        - frame (:obj:`numpy.ndarray`): [H * W * C]
    '''
    gym_env = env._env
    if hasattr(gym_env, 'sim'):
        # mujoco: mujoco frame is unside-down by default
        return gym_env.sim.render(camera_name='track', height=128, width=128)[::-1]
    else:
        # other envs
        return gym_env.render(mode=render_mode)


def fps(env_manager: "BaseEnvManager") -> "int":
    '''
    Overview:
        Render the environment's fps.
    Arguments:
        - env (:obj:`BaseEnvManager`): DI-engine env manager instance.
    Returns:
        - fps (:obj:`int`).
    '''
    try:
        # env_ref is a ding gym environment
        gym_env = env_manager.env_ref._env
        if hasattr(gym_env, 'model'):
            # mujoco
            fps = 1 / gym_env.model.opt.timestep
        elif hasattr(gym_env, 'env') and 'video.frames_per_second' in gym_env.env.metadata.keys():
            # classic control
            fps = gym_env.env.metadata['video.frames_per_second']
        else:
            # atari and other envs
            fps = 30
        return fps
    except:
        return 30
