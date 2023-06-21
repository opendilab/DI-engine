import gym
from ditk import logging

from ding.config import compile_config
from ding.data import DequeBuffer
from ding.envs import BaseEnvManagerV2, DingEnvWrapper
from ding.framework import ding_init, task
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import CkptSaver, OffPolicyLearner, StepCollector, data_pusher, eps_greedy_handler, \
    interaction_evaluator, online_logger
from ding.model import DQN
from ding.policy import DQNPolicy
from ding.utils import set_pkg_seed
from dizoo.cliffwalking.config.cliffwalking_dqn_config import create_config, main_config
from dizoo.cliffwalking.envs.cliffwalking_env import CliffWalkingEnv


def main():
    filename = '{}/log.txt'.format(main_config.exp_name)
    logging.getLogger(with_files=[filename]).setLevel(logging.INFO)

    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    ding_init(cfg)

    collector_env = BaseEnvManagerV2(
        env_fn=[lambda: CliffWalkingEnv(cfg.env) for _ in range(cfg.env.collector_env_num)], cfg=cfg.env.manager
    )
    evaluator_env = BaseEnvManagerV2(
        env_fn=[lambda: CliffWalkingEnv(cfg.env) for _ in range(cfg.env.collector_env_num)], cfg=cfg.env.manager
    )

    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

    model = DQN(**cfg.policy.model)
    buffer = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
    policy = DQNPolicy(cfg.policy, model=model)

    with task.start(async_mode=False, ctx=OnlineRLContext()):
        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(eps_greedy_handler(cfg))
        task.use(StepCollector(cfg, policy.collect_mode, collector_env))
        task.use(data_pusher(cfg, buffer))
        task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer))
        task.use(online_logger(train_show_freq=10))
        task.use(CkptSaver(policy, cfg.exp_name, train_freq=1000))
        task.run()


if __name__ == '__main__':
    main()
