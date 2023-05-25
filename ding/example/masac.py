import gym
from ditk import logging
from ding.model import MAQAC
from ding.policy import SACDiscretePolicy
from ding.envs import DingEnvWrapper, BaseEnvManagerV2
from ding.data import DequeBuffer
from ding.config import compile_config
from ding.framework import task, ding_init
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import OffPolicyLearner, StepCollector, interaction_evaluator, CkptSaver, \
    data_pusher, online_logger, termination_checker, eps_greedy_handler
from ding.utils import set_pkg_seed
from dizoo.petting_zoo.config.ptz_simple_spread_masac_config import main_config, create_config
from dizoo.petting_zoo.envs.petting_zoo_simple_spread_env import PettingZooEnv


def main():
    logging.getLogger().setLevel(logging.INFO)
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    ding_init(cfg)
    with task.start(async_mode=False, ctx=OnlineRLContext()):
        collector_env = BaseEnvManagerV2(
            env_fn=[lambda: PettingZooEnv(cfg.env) for _ in range(cfg.env.collector_env_num)], cfg=cfg.env.manager
        )
        evaluator_env = BaseEnvManagerV2(
            env_fn=[lambda: PettingZooEnv(cfg.env) for _ in range(cfg.env.evaluator_env_num)], cfg=cfg.env.manager
        )

        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

        model = MAQAC(**cfg.policy.model)
        buffer_ = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
        policy = SACDiscretePolicy(cfg.policy, model=model)

        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(eps_greedy_handler(cfg))
        task.use(
            StepCollector(cfg, policy.collect_mode, collector_env, random_collect_size=cfg.policy.random_collect_size)
        )
        task.use(data_pusher(cfg, buffer_))
        task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer_, log_freq=100))
        task.use(CkptSaver(policy, cfg.exp_name, train_freq=1000))
        task.use(online_logger(train_show_freq=10))
        task.use(termination_checker(max_env_step=int(1e6)))
        task.run()


if __name__ == "__main__":
    main()
