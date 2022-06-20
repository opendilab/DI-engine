from tabnanny import check
from typing import Any, Callable, List, Tuple
import numpy as np
from collections.abc import Sequence
from easydict import EasyDict

from ding.envs.env import BaseEnv, BaseEnvTimestep
from ding.envs.env.tests import DemoEnv
# from dizoo.atari.envs import AtariEnv


def check_space_dtype(env: BaseEnv) -> None:
    print("== 0. Test obs/act/rew space's dtype")
    env.reset()
    for name, space in zip(['obs', 'act', 'rew'], [env.observation_space, env.action_space, env.reward_space]):
        if 'float' in repr(space.dtype):
            assert space.dtype == np.float32, "If float, then must be np.float32, but get {} for {} space".format(
                space.dtype, name
            )
        if 'int' in repr(space.dtype):
            assert space.dtype == np.int64, "If int, then must be np.int64, but get {} for {} space".format(
                space.dtype, name
            )


# Util function
def check_array_space(ndarray, space, name) -> bool:
    if isinstance(ndarray, np.ndarray):
        # print("{}'s type should be np.ndarray".format(name))
        assert ndarray.dtype == space.dtype, "{}'s dtype is {}, but requires {}".format(
            name, ndarray.dtype, space.dtype
        )
        assert ndarray.shape == space.shape, "{}'s shape is {}, but requires {}".format(
            name, ndarray.shape, space.shape
        )
        assert (space.low <= ndarray).all() and (ndarray <= space.high).all(
        ), "{}'s value is {}, but requires in range ({},{})".format(name, ndarray, space.low, space.high)
    elif isinstance(ndarray, Sequence):
        for i in range(len(ndarray)):
            try:
                check_array_space(ndarray[i], space[i], name)
            except AssertionError as e:
                print("The following  error happens at {}-th index".format(i))
                raise e
    elif isinstance(ndarray, dict):
        for k in ndarray.keys():
            try:
                check_array_space(ndarray[k], space[k], name)
            except AssertionError as e:
                print("The following  error happens at key {}".format(k))
                raise e
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
        if done:
            assert 'final_eval_reward' in info, "info dict should have 'final_eval_reward' key."
            done_times += 1
            _ = env.reset()
        if done_times == 3:
            break


# Util function
def check_different_memory(array1, array2, step_times) -> None:
    assert type(array1) == type(
        array2
    ), "In step times {}, obs_last_frame({}) and obs_this_frame({}) are not of the same type".format(
        step_times, type(array1), type(array2)
    )
    if isinstance(array1, np.ndarray):
        assert id(array1) != id(
            array2
        ), "In step times {}, obs_last_frame and obs_this_frame are the same np.ndarray".format(step_times)
    elif isinstance(array1, Sequence):
        assert len(array1) == len(
            array2
        ), "In step times {}, obs_last_frame({}) and obs_this_frame({}) have different sequence lengths".format(
            step_times, len(array1), len(array2)
        )
        for i in range(len(array1)):
            try:
                check_different_memory(array1[i], array2[i], step_times)
            except AssertionError as e:
                print("The following error happens at {}-th index".format(i))
                raise e
    elif isinstance(array1, dict):
        assert array1.keys() == array2.keys(), "In step times {}, obs_last_frame({}) and obs_this_frame({}) have \
                different dict keys".format(step_times, array1.keys(), array2.keys())
        for k in array1.keys():
            try:
                check_different_memory(array1[k], array2[k], step_times)
            except AssertionError as e:
                print("The following  error happens at key {}".format(k))
                raise e
    else:
        raise TypeError(
            "Input array should be np.ndarray or list/dict of np.ndarray, but found {} and {}".format(
                type(array1), type(array2)
            )
        )


def check_obs_deepcopy(env: BaseEnv) -> None:

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


def check_all(env: BaseEnv) -> None:
    check_space_dtype(env)
    check_reset(env)
    check_step(env)
    check_obs_deepcopy(env)


def demonstrate_correct_procedure(env_fn: Callable) -> None:
    print('== 4. Demonstrate the correct procudures')
    done_times = 0
    # Init the env.
    env = env_fn({})
    # Lazy init. The real env is not initialized until `reset` method is called
    assert not hasattr(env, "_env")
    # Must set seed before `reset` method is called.
    env.seed(4)
    assert env._seed == 4
    # Reset the env. The real env is initialized here.
    obs = env.reset()
    while True:
        # Using the policy to get the action from obs. But here we use `random_action` instead.
        action = env.random_action()
        obs, rew, done, info = env.step(action)
        if done:
            assert 'final_eval_reward' in info
            done_times += 1
            obs = env.reset()
            # Seed will not change unless `seed` method is called again.
            assert env._seed == 4
        if done_times == 3:
            break


if __name__ == "__main__":
    '''
    # Moethods `check_*` are for user to check whether their implemented env obeys DI-engine's rules.
    # You can replace `AtariEnv` with your own env.
    atari_env = AtariEnv(EasyDict(env_id='PongNoFrameskip-v4', frame_stack=4, is_train=False))
    check_reset(atari_env)
    check_step(atari_env)
    check_obs_deepcopy(atari_env)
    '''
    # Method `demonstrate_correct_procudure` is to demonstrate the correct procedure to
    # use an env to generate trajectories.
    # You can check whether your env's design is similar to `DemoEnv`
    demonstrate_correct_procedure(DemoEnv)
