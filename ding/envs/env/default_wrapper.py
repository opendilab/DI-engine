from easydict import EasyDict
from typing import Optional, List
import copy

final_eval_reward_wrapper = EasyDict(type='final_eval_reward')


def get_default_wrappers(env_wrapper_name: str, env_id: Optional[str] = None) -> List[dict]:
    if env_wrapper_name == 'mujoco_default':
        return [
            EasyDict(type='delay_reward', kwargs=dict(delay_reward_step=3)),
            copy.deepcopy(final_eval_reward_wrapper),
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
        wrapper_list.append(EasyDict(type='clip_reward'))
        wrapper_list.append(EasyDict(type='frame_stack', kwargs=dict(n_frames=4)))
        wrapper_list.append(copy.deepcopy(final_eval_reward_wrapper))
        return wrapper_list
    elif env_wrapper_name == 'gym_hybrid_default':
        return [
            EasyDict(type='gym_hybrid_dict_action'),
            copy.deepcopy(final_eval_reward_wrapper),
        ]
    elif env_wrapper_name == 'default':
        return [copy.deepcopy(final_eval_reward_wrapper)]
    else:
        raise NotImplementedError()
