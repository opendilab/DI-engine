import gym
from ditk import logging
from ding.model import MAVAC
from ding.policy import PPOPolicy
from ding.envs import DingEnvWrapper, SubprocessEnvManagerV2
from ding.data import DequeBuffer
from ding.config import compile_config
from ding.framework import task, ding_init
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import multistep_trainer, StepCollector, interaction_evaluator, CkptSaver, \
    gae_estimator, online_logger, termination_checker
from ding.utils import set_pkg_seed
from dizoo.petting_zoo.config.ptz_simple_spread_mappo_config import main_config, create_config
from dizoo.petting_zoo.envs.petting_zoo_simple_spread_env import PettingZooEnv


def main():
    logging.getLogger().setLevel(logging.INFO)
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    ding_init(cfg)
    with task.start(async_mode=False, ctx=OnlineRLContext()):
        collector_env = SubprocessEnvManagerV2(
            env_fn=[lambda: PettingZooEnv(cfg.env) for _ in range(cfg.env.collector_env_num)], cfg=cfg.env.manager
        )
        evaluator_env = SubprocessEnvManagerV2(
            env_fn=[lambda: PettingZooEnv(cfg.env) for _ in range(cfg.env.evaluator_env_num)], cfg=cfg.env.manager
        )

        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

        model = MAVAC(**cfg.policy.model)
        policy = PPOPolicy(cfg.policy, model=model)

        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(StepCollector(cfg, policy.collect_mode, collector_env))
        task.use(gae_estimator(cfg, policy.collect_mode))
        task.use(multistep_trainer(policy.learn_mode, log_freq=100))
        task.use(CkptSaver(policy, cfg.exp_name, train_freq=1000))
        task.use(online_logger(train_show_freq=10))
        task.use(termination_checker(max_env_step=int(1e6)))
        task.run()


if __name__ == "__main__":
    main()
