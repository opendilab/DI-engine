from tabnanny import check
from typing import Any, List, Tuple
import numpy as np
from collections import Sequence
from easydict import EasyDict

from ding.envs.env import BaseEnv, BaseEnvTimestep
from dizoo.classic_control.cartpole.envs import CartPoleEnv
from dizoo.atari.envs import AtariEnv


def check_array_space(ndarray, space, name) -> bool:
    if isinstance(ndarray, np.ndarray):
        # print("{}'s type should be np.ndarray".format(name))
        if ndarray.dtype != space.dtype:
            print("{}'s dtype is {}, but requires {}".format(name, ndarray.dtype, space.dtype))
        if ndarray.shape != space.shape:
            print("{}'s shape is {}, but requires {}".format(name, ndarray.shape, space.shape))
        if not (space.low <= ndarray).all() or not (ndarray <= space.high).all():
            print("{}'s value is {}, but requires in range ({},{})".format(name, ndarray, space.low, space.high))
    elif isinstance(ndarray, Sequence):
        for i in range(len(ndarray)):
            check_array_space(ndarray[i], space[i], name)
            # if not check_array_space(ndarray[i], space[i], name):
            #     print("Aformentioned error happens at {}-th index".format(i))
    elif isinstance(ndarray, dict):
        for k in ndarray.keys():
            check_array_space(ndarray[k], space[k], name)
            # if not check_array_space(ndarray[k], space[k], name):
            #     print("Aformentioned error happens at {} key".format(k))
    else:
        raise TypeError(
            "Input array should be np.ndarray or sequence/dict of np.ndarray, but found {}".format(type(ndarray))
        )


def check_reset(env: BaseEnv) -> None:
    print('== 1. Test reset method')
    obs = env.reset()
    check_array_space(obs, env.observation_space, 'obs')


def check_step(env: BaseEnv) -> None:
    done_times = 0
    print('== 2. Test step method')
    _ = env.reset()
    if hasattr(env, "random_action"):
        random_action = env.random_action()
    else:
        random_action = env.action_space.sample()
    while True:
        obs, rew, done, info = env.step(random_action)
        for ndarray, space, name in zip([obs, rew], [env.observation_space, env.reward_space], ['obs', 'rew']):
            check_array_space(ndarray, space, name)
        if done and 'final_eval_reward' not in info:
            print("info dict should have 'final_eval_reward' key.")
        if done:
            done_times += 1
            _ = env.reset()
        if done_times == 3:
            break


def check_obs_deepcopy(env: BaseEnv) -> None:

    def check_different_memory(array1, array2, step_times) -> bool:
        if type(array1) != type(array2):
            print(
                "In step times {}, obs_last_frame({}) and obs_this_frame({}) are not of the same type".format(
                    step_times, type(array1), type(array2)
                )
            )
            return False
        if isinstance(array1, np.ndarray):
            if id(array1) == id(array2):
                print("In step times {}, obs_last_frame and obs_this_frame are the same np.ndarray".format(step_times))
                return False
        elif isinstance(array1, Sequence):
            if len(array1) != len(array2):
                print(
                    "In step times {}, obs_last_frame({}) and obs_this_frame({}) have different sequence lengths".
                    format(step_times, len(array1), len(array2))
                )
                return False
            for i in range(len(array1)):
                if not check_different_memory(array1[i], array2[i]):
                    print("Aformentioned error happens at {}-th index".format(i))
                    return False
        elif isinstance(array1, dict):
            if array1.keys != array2.keys():
                print(
                    "In step times {}, obs_last_frame({}) and obs_this_frame({}) have different dict keys".format(
                        step_times, array1.keys(), array2.keys()
                    )
                )
                return False
            for k in array1.keys():
                if not check_different_memory(array1[k], array2[k]):
                    print("Aformentioned error happens at {} key".format(k))
                    return False
        else:
            raise TypeError(
                "Input array should be np.ndarray or list/dict of np.ndarray, but found {} and {}".format(
                    type(array1), type(array2)
                )
            )
        return True

    step_times = 0
    print('== 3. Test observation deepcopy')
    obs_1 = env.reset()
    if hasattr(env, "random_action"):
        random_action = env.random_action()
    else:
        random_action = env.action_space.sample()
    while True:
        step_times += 1
        obs_2, _, done, _ = env.step(random_action)
        check_different_memory(obs_1, obs_2, step_times)
        obs_1 = obs_2
        if done:
            break


if __name__ == "__main__":
    # cartpole_env = CartPoleEnv({})
    cartpole_env = AtariEnv(EasyDict(env_id='PongNoFrameskip-v4', frame_stack=4, is_train=False))
    check_reset(cartpole_env)
    check_step(cartpole_env)
    check_obs_deepcopy(cartpole_env)
