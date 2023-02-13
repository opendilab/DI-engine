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
    eps_greedy_handler, CkptSaver, context_exchanger, model_exchanger, termination_checker, nstep_reward_enhancer, \
    online_logger
from ding.utils import set_pkg_seed
from dizoo.atari.envs.atari_env import AtariEnv
from dizoo.atari.config.serial.pong.pong_dqn_config import main_config, create_config


def main():
    logging.getLogger().setLevel(logging.INFO)
    main_config.exp_name = 'pong_dqn_seed0_dist_rdma'
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    ding_init(cfg)
    with task.start(async_mode=False, ctx=OnlineRLContext()):
        assert task.router.is_active, "Please execute this script with ditask! See note in the header."

        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

        model = DQN(**cfg.policy.model)
        policy = DQNPolicy(cfg.policy, model=model)

        if 'learner' in task.router.labels:
            logging.info("Learner running on node {}".format(task.router.node_id))
            buffer_ = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
            task.use(
                context_exchanger(
                    send_keys=["train_iter"],
                    recv_keys=["trajectories", "episodes", "env_step", "env_episode"],
                    skip_n_iter=0
                )
            )
            task.use(model_exchanger(model, is_learner=True))
            task.use(nstep_reward_enhancer(cfg))
            task.use(data_pusher(cfg, buffer_))
            task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer_))
            task.use(CkptSaver(policy, cfg.exp_name, train_freq=1000))

        elif 'collector' in task.router.labels:
            logging.info("Collector running on node {}".format(task.router.node_id))
            collector_cfg = deepcopy(cfg.env)
            collector_cfg.is_train = True
            collector_env = SubprocessEnvManagerV2(
                env_fn=[lambda: AtariEnv(collector_cfg) for _ in range(cfg.env.collector_env_num)], cfg=cfg.env.manager
            )
            task.use(
                context_exchanger(
                    send_keys=["trajectories", "episodes", "env_step", "env_episode"],
                    recv_keys=["train_iter"],
                    skip_n_iter=1
                )
            )
            task.use(model_exchanger(model, is_learner=False))
            task.use(eps_greedy_handler(cfg))
            task.use(StepCollector(cfg, policy.collect_mode, collector_env))
            task.use(termination_checker(max_env_step=int(1e7)))
        else:
            raise KeyError("invalid router labels: {}".format(task.router.labels))

        task.run()


if __name__ == "__main__":
    main()
