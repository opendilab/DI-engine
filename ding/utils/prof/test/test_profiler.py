import pytest
import os
import random
import torch
import copy

from ding.envs import DingEnvWrapper, SubprocessEnvManagerV2
from time import sleep
from ding.utils.prof.di_profiler import get_profiler, ProfileType
from ding.framework import task, Context, Parallel, OnlineRLContext
from ding.model import DQN
from ding.policy import DQNPolicy
from ding.framework.middleware.tests import MockPolicy, MockEnv, CONFIG
from ditk import logging
from dizoo.atari.envs.atari_env import AtariEnv
from ding.framework.middleware import StepCollector, EpisodeCollector, eps_greedy_handler, termination_checker
from dizoo.atari.config.serial.pong.pong_dqn_config import main_config, create_config
from ding.config import compile_config


@pytest.mark.unittest
def test_layer_profiler():

    main_config.exp_name = 'pong_dqn_seed0_dist_rdma'
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    model = DQN(**cfg.policy.model)
    policy = DQNPolicy(cfg.policy, model=model)

    with task.start(async_mode=False, model=model, ctx=OnlineRLContext(), prof_type=ProfileType.LAYER,
                    prof_trace_path="/mnt/cache/wangguoteng.p/DI-engine"):
        logging.info("Collector running on node")
        collector_cfg = copy.deepcopy(cfg.env)
        collector_cfg.is_train = True
        collector_env = SubprocessEnvManagerV2(
            env_fn=[lambda: AtariEnv(collector_cfg) for _ in range(cfg.env.collector_env_num)], cfg=cfg.env.manager
        )
        task.use(eps_greedy_handler(cfg))
        task.use(StepCollector(cfg, policy.collect_mode, collector_env))
        task.use(termination_checker(max_env_step=int(1e7)))
        task.run(max_step=1)


if __name__ == "__main__":
    test_layer_profiler()
