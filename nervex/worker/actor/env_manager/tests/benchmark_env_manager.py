import time
import numpy as np
import pytest
import torch
import random
import sys
from easydict import EasyDict
from nervex.worker.actor.env_manager.vec_env_manager import SubprocessEnvManager, SyncSubprocessEnvManager
from nervex.worker.actor.env_manager.base_env_manager import BaseEnvManager
from nervex.envs import get_subprocess_env_setting
import cloudpickle
from nervex.model.actor_critic.value_ac import ValueAC, ConvValueAC


def timer(fn):

    def time_func(*args, **kwargs):
        start_time = time.time()
        ret = fn(*args, **kwargs)
        print("%s runs for %.2f sec" % (fn, time.time() - start_time))
        return ret

    return time_func


class Timer(object):

    def __init__(self, block_name=""):
        self.block_name = block_name

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, type, value, traceback):
        end_time = time.time()
        print(self.block_name, "%.2f sec" % (end_time - self.start_time))


class BenchmarkEnvManager(object):

    def __init__(self, env_fn, env_cfg, env_num, test_max_sample=1e2, seed=2021):
        self.env_fn = env_fn
        self.env_cfg = env_cfg
        self.env_num = env_num
        self.obs_dim = actor_env_cfg[0].obs_dim
        self.action_dim = actor_env_cfg[0].action_dim
        self.continous = actor_env_cfg[0].continous
        self.test_max_sample = int(test_max_sample)
        self.seed = seed
        # self.model = ConvValueAC(self.obs_dim, self.action_dim, embedding_dim=64, continous=self.continous)

    def baseline_rand_action(self):
        with Timer("launch BaseEnvManager: "):
            env_manager = BaseEnvManager(env_fn=self.env_fn, env_cfg=[self.env_cfg[0]], env_num=1)
            env_manager.seed([self.seed])
            env_manager.launch()

        with Timer("baseline_rand_action:"):
            sample_num = 0
            while sample_num < self.test_max_sample:
                obses = env_manager.next_obs
                # dummy_obses = torch.randn([len(obses)] + self.obs_dim)
                # _ = self.model(dummy_obses, mode='compute_action_value')
                actions = {}
                for env_id in obses.keys():
                    if self.continous:
                        actions[env_id] = torch.rand(self.action_dim)
                    else:
                        actions[env_id] = torch.randint(0, self.action_dim, (1, ))
                timesteps = env_manager.step(actions)
                sample_num += len(timesteps)

        env_manager.close()

    def sync_baseline_rand_action(self):
        with Timer("launch BaseEnvManager: "):
            env_manager = BaseEnvManager(env_fn=self.env_fn, env_cfg=self.env_cfg, env_num=len(self.env_cfg))
            env_manager.seed([self.seed])
            env_manager.launch()

        with Timer("sync_baseline_rand_action:"):
            sample_num = 0
            while sample_num < self.test_max_sample:
                obses = env_manager.next_obs
                # dummy_obses = torch.randn([len(obses)] + self.obs_dim)
                # _ = self.model(dummy_obses, mode='compute_action_value')
                actions = {}
                for env_id in obses.keys():
                    if self.continous:
                        actions[env_id] = torch.rand(self.action_dim)
                    else:
                        actions[env_id] = torch.randint(0, self.action_dim, (1, ))
                timesteps = env_manager.step(actions)
                sample_num += len(timesteps)

        env_manager.close()

    def test_rand_action(self):
        with Timer("launch SubprocessEnvManager: "):
            env_manager = SubprocessEnvManager(env_fn=self.env_fn, env_cfg=self.env_cfg, env_num=len(self.env_cfg))
            env_manager.seed([self.seed for _ in range(env_manager.env_num)])
            env_manager.launch()

        with Timer("test_rand_action: "):
            sample_num = 0
            while sample_num < self.test_max_sample:
                obses = env_manager.next_obs
                # dummy_obses = torch.randn([len(obses)] + self.obs_dim)
                # _ = self.model(dummy_obses, mode='compute_action_value')
                actions = {}
                for env_id in obses.keys():
                    if self.continous:
                        actions[env_id] = torch.rand(self.action_dim)
                    else:
                        actions[env_id] = torch.randint(0, self.action_dim, (1, ))
                timesteps = env_manager.step(actions)
                sample_num += len(timesteps)

        env_manager.close()


if __name__ == "__main__":
    import cProfile, pstats, io

    # ## Pendulum
    # env_cfg = EasyDict({
    # "env_manager_type": 'base',
    # "import_names": ['app_zoo.classic_control.pendulum.envs.pendulum_env'],
    # "env_type": 'pendulum',
    # "actor_env_num": 12,
    # "evaluator_env_num": 1,
    # "obs_dim": 3,
    # "action_dim": 1,
    # "use_act_scale": True,
    # "continous": True
    # })

    # # Cartpole
    # env_cfg = EasyDict({
    # "env_manager_type": 'base',
    # "import_names": ['app_zoo.classic_control.cartpole.envs.cartpole_env'],
    # "env_type": 'cartpole',
    # "actor_env_num": 12,
    # "evaluator_env_num": 1,
    # "obs_dim": 4,
    # "action_dim": 2,
    # "continous": False
    # })

    # # pong
    # env_cfg = EasyDict(
    #     {
    #         "import_names": ['app_zoo.atari.envs.atari_env'],
    #         "env_type": 'atari',
    #         "env_id": 'PongNoFrameskip-v4',
    #         "frame_stack": 4,
    #         "actor_env_num": 12,
    #         "evaluator_env_num": 3,
    #         "action_dim": 3,
    #         "obs_dim": [4, 84, 84],
    #         "continous": False
    #     }
    # )

    # # qbert
    # # 1e4 30s, 13s, 9s
    # env_cfg = EasyDict(
    #     {
    #         "import_names": ['app_zoo.atari.envs.atari_env'],
    #         "env_type": 'atari',
    #         "env_id": 'QbertNoFrameskip-v4',
    #         "frame_stack": 4,
    #         "actor_env_num": 12,
    #         "evaluator_env_num": 1,
    #         "action_dim": 6,
    #         "obs_dim": [4, 84, 84],
    #         "continous": False
    #     }
    # )

    # # invader
    # env_cfg = EasyDict(
    #     {
    #         "import_names": ['app_zoo.atari.envs.atari_env'],
    #         "env_type": 'atari',
    #         "env_id": 'SpaceInvadersNoFrameskip-v4',
    #         "frame_stack": 4,
    #         "actor_env_num": 12,
    #         "evaluator_env_num": 1,
    #         "action_dim": 6,
    #         "obs_dim": [4, 84, 84],
    #         "continous": False
    #     }
    # )

    # Halfcheetah
    env_cfg = EasyDict(
        {
            "import_names": ['app_zoo.mujoco.envs.mujoco_env'],
            "env_type": 'mujoco',
            "env_id": 'HalfCheetah-v2',
            "frame_stack": 4,
            "actor_env_num": 12,
            "evaluator_env_num": 3,
            "action_dim": 6,
            "obs_dim": 17,
            "use_act_scale": True,
            "is_train": True,
            "continous": True
        }
    )

    MAX_TEST_TIME = 1e5
    TEST_SEED = 2021

    env_fn, actor_env_cfg, evaluator_env_cfg = get_subprocess_env_setting(env_cfg)
    benchmark = BenchmarkEnvManager(
        env_fn=env_fn, env_cfg=actor_env_cfg, env_num=len(actor_env_cfg), test_max_sample=MAX_TEST_TIME, seed=TEST_SEED
    )

    # benchmark.baseline_rand_action()
    # benchmark.sync_baseline_rand_action()
    benchmark.test_rand_action()

    # pr = cProfile.Profile()
    # pr.enable()
    # benchmark.test_rand_action()
    # pr.disable()
    # s = io.StringIO()
    # sortby = "cumtime"
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats(20)
    # print(s.getvalue())
    # pr.dump_stats("pipeline.prof")
