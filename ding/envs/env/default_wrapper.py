from easydict import EasyDict
from typing import Optional, List
import copy

eval_episode_return_wrapper = EasyDict(type='eval_episode_return')


def get_default_wrappers(env_wrapper_name: str, env_id: Optional[str] = None, caller: str = 'collector') -> List[dict]:
    """
    Overview:
        Get default wrappers for different environments used in ``DingEnvWrapper``.
    Arguments:
        - env_wrapper_name (:obj:`str`): The name of the environment wrapper.
        - env_id (:obj:`Optional[str]`): The id of the specific environment, such as ``PongNoFrameskip-v4``.
        - caller (:obj:`str`): The caller of the environment, including ``collector`` or ``evaluator``. Different \
            caller may need different wrappers.
    Returns:
        - wrapper_list (:obj:`List[dict]`): The list of wrappers, each element is a config of the concrete wrapper.
    Raises:
        - NotImplementedError: ``env_wrapper_name`` is not in ``['mujoco_default', 'atari_default', \
            'gym_hybrid_default', 'default']``
    """
    assert caller == 'collector' or 'evaluator', caller
    if env_wrapper_name == 'mujoco_default':
        return [
            copy.deepcopy(eval_episode_return_wrapper),
        ]
    elif env_wrapper_name == 'atari_default':
        wrapper_list = []
        wrapper_list.append(EasyDict(type='noop_reset', kwargs=dict(noop_max=30)))
        wrapper_list.append(EasyDict(type='max_and_skip', kwargs=dict(skip=4)))
        wrapper_list.append(EasyDict(type='episodic_life'))
        if env_id is not None:
            if 'Pong' in env_id or 'Qbert' in env_id or 'SpaceInvader' in env_id or 'Montezuma' in env_id:
                wrapper_list.append(EasyDict(type='fire_reset'))
        wrapper_list.append(EasyDict(type='warp_frame'))
        wrapper_list.append(EasyDict(type='scaled_float_frame'))
        if caller == 'collector':
            wrapper_list.append(EasyDict(type='clip_reward'))
        wrapper_list.append(EasyDict(type='frame_stack', kwargs=dict(n_frames=4)))
        wrapper_list.append(copy.deepcopy(eval_episode_return_wrapper))
        return wrapper_list
    elif env_wrapper_name == 'gym_hybrid_default':
        return [
            EasyDict(type='gym_hybrid_dict_action'),
            copy.deepcopy(eval_episode_return_wrapper),
        ]
    elif env_wrapper_name == 'default':
        return [copy.deepcopy(eval_episode_return_wrapper)]
    else:
        raise NotImplementedError("not supported env_wrapper_name: {}".format(env_wrapper_name))
