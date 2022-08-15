from copy import deepcopy
from ditk import logging
from ding.model import DQN
from ding.policy import DQNPolicy
from ding.envs import DingEnvWrapper, SubprocessEnvManagerV2
from ding.data import DequeBuffer
from ding.config import compile_config
from ding.framework import task, ding_init
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import OffPolicyLearner, StepCollector, interaction_evaluator, data_pusher, \
    eps_greedy_handler, CkptSaver, ContextExchanger, ModelExchanger, nstep_reward_enhancer, termination_checker
from ding.utils import set_pkg_seed
from dizoo.atari.envs.atari_env import AtariEnv
from dizoo.atari.config.serial.pong.pong_dqn_config import main_config, create_config
from ding.utils import EasyTimer
import os
import time


def main():
    logger = logging.getLogger().setLevel(logging.DEBUG)
    main_config.exp_name = 'pong_dqn_seed0_dist_rdma'
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    ding_init(cfg)
    with task.start(async_mode=False, ctx=OnlineRLContext()):
        assert task.router.is_active, "Please execute this script with ditask! See note in the header."

        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

        model = DQN(**cfg.policy.model)

        # Consider the case with multiple processes
        if task.router.is_active:
            # You can use labels to distinguish between workers with different roles,
            # here we use node_id to distinguish.
            if task.router.node_id == 0:
                task.add_role(task.role.LEARNER)
            else:
                task.add_role(task.role.COLLECTOR)

        logging.debug("label {}".format(task.router.labels))
        logging.debug("task role {}".format(task._roles))

        if 'learner' in task.router.labels:
            policy = DQNPolicy(cfg.policy, model=model)
            logging.info("Learner running on node {}".format(task.router.node_id))
            buffer_ = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
            task.use(ContextExchanger(skip_n_iter=0))
            task.use(ModelExchanger(model))
            task.use(nstep_reward_enhancer(cfg))
            task.use(data_pusher(cfg, buffer_))
            task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer_))
            task.use(CkptSaver(cfg, policy, train_freq=1000))

        elif 'collector' in task.router.labels:
            policy = DQNPolicy(cfg.policy, model=model)
            logging.info("Collector running on node {}".format(task.router.node_id))
            collector_cfg = deepcopy(cfg.env)
            collector_cfg.is_train = True
            logging.info(cfg.env.manager)
            logging.info(type(cfg.env.manager))
            # task.router.judge_use_cuda_shm(cfg)
            logging.debug("cuda_shared_memory {}".format(cfg.env.manager.cuda_shared_memory))
            collector_env = SubprocessEnvManagerV2(
                env_fn=[lambda: AtariEnv(collector_cfg) for _ in range(cfg.env.collector_env_num)], cfg=cfg.env.manager
            )
            task.use(ContextExchanger(skip_n_iter=1))
            task.use(ModelExchanger(model))
            task.use(eps_greedy_handler(cfg))
            task.use(StepCollector(cfg, policy.collect_mode, collector_env))
            task.use(termination_checker(max_env_step=int(1e7)))
        else:
            raise KeyError("invalid router labels: {}".format(task.router.labels))

        start_time = task.run(max_step=100)
        end_time = time.time()
        logging.debug("atari iter 99 use {:.4f} s,".format(end_time - start_time))


if __name__ == "__main__":
    main()
