from copy import deepcopy
from ditk import logging
from ding.model import VAC
from ding.policy import PPOPolicy
from ding.envs import DingEnvWrapper, SubprocessEnvManagerV2
from ding.data import DequeBuffer
from ding.config import compile_config
from ding.framework import task, ding_init
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import multistep_trainer, StepCollector, interaction_evaluator, CkptSaver, \
    gae_estimator, ddp_termination_checker, online_logger
from ding.utils import set_pkg_seed, DistContext, get_rank, get_world_size
from dizoo.atari.envs.atari_env import AtariEnv
from dizoo.atari.config.serial.pong.pong_onppo_config import main_config, create_config


def main():
    logging.getLogger().setLevel(logging.INFO)
    with DistContext():
        rank, world_size = get_rank(), get_world_size()
        main_config.example = 'pong_ppo_seed0_ddp_avgsplit'
        main_config.policy.multi_gpu = True
        main_config.policy.learn.batch_size = main_config.policy.learn.batch_size // world_size
        main_config.policy.collect.n_sample = main_config.policy.collect.n_sample // world_size
        cfg = compile_config(main_config, create_cfg=create_config, auto=True)
        ding_init(cfg)
        with task.start(async_mode=False, ctx=OnlineRLContext()):
            collector_cfg = deepcopy(cfg.env)
            collector_cfg.is_train = True
            evaluator_cfg = deepcopy(cfg.env)
            evaluator_cfg.is_train = False
            collector_env = SubprocessEnvManagerV2(
                env_fn=[lambda: AtariEnv(collector_cfg) for _ in range(cfg.env.collector_env_num)], cfg=cfg.env.manager
            )
            evaluator_env = SubprocessEnvManagerV2(
                env_fn=[lambda: AtariEnv(evaluator_cfg) for _ in range(cfg.env.evaluator_env_num)], cfg=cfg.env.manager
            )

            set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

            model = VAC(**cfg.policy.model)
            policy = PPOPolicy(cfg.policy, model=model)

            if rank == 0:
                task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
            task.use(StepCollector(cfg, policy.collect_mode, collector_env))
            task.use(gae_estimator(cfg, policy.collect_mode))
            task.use(multistep_trainer(cfg, policy.learn_mode))
            if rank == 0:
                task.use(CkptSaver(policy, cfg.exp_name, train_freq=1000))
            task.use(ddp_termination_checker(max_env_step=int(1e7), rank=rank))
            task.run()


if __name__ == "__main__":
    main()
