import gym
from ditk import logging
from ding.model import VAC
from ding.policy import IMPALAPolicy
from ding.envs import SubprocessEnvManagerV2
from ding.data import DequeBuffer
from ding.config import compile_config
from ding.framework import task, ding_init
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import OffPolicyLearner, StepCollector, interaction_evaluator, data_pusher, \
    CkptSaver, online_logger, termination_checker
from ding.utils import set_pkg_seed
from dizoo.box2d.lunarlander.config.lunarlander_impala_config import main_config, create_config
from dizoo.box2d.lunarlander.envs import LunarLanderEnv


def main():
    logging.getLogger().setLevel(logging.INFO)
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    ding_init(cfg)
    with task.start(async_mode=False, ctx=OnlineRLContext()):
        collector_env = SubprocessEnvManagerV2(
            env_fn=[lambda: LunarLanderEnv(cfg.env) for _ in range(cfg.env.collector_env_num)], cfg=cfg.env.manager
        )
        evaluator_env = SubprocessEnvManagerV2(
            env_fn=[lambda: LunarLanderEnv(cfg.env) for _ in range(cfg.env.evaluator_env_num)], cfg=cfg.env.manager
        )
        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

        model = VAC(**cfg.policy.model)
        buffer_ = DequeBuffer(
            size=cfg.policy.other.replay_buffer.replay_buffer_size, sliced=cfg.policy.other.replay_buffer.sliced
        )
        policy = IMPALAPolicy(cfg.policy, model=model)

        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(StepCollector(cfg, policy.collect_mode, collector_env, random_collect_size=1024))
        task.use(data_pusher(cfg, buffer_, group_by_env=True))
        task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer_))
        task.use(online_logger(train_show_freq=300))
        task.use(CkptSaver(policy, cfg.exp_name, train_freq=10000))
        task.use(termination_checker(max_env_step=2e6))
        task.run()


if __name__ == "__main__":
    main()
