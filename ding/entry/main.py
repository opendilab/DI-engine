"""
Main entry
"""
from functools import partial
from ding.model import QAC
from ding.utils import set_pkg_seed
from ding.envs import BaseEnvManager, get_vec_env_setting
from ding.config import compile_config
from ding.policy import SACPolicy

from ding.framework.middleware import basic_collector, basic_evaluator, basic_learner
from ding.framework import Task
from dizoo.classic_control.pendulum.config.pendulum_sac_config import main_config, create_config
from ding.worker.buffer import DequeBuffer


def main(cfg, model):
    with Task(async_mode=False) as task:
        env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)

        collector_env = BaseEnvManager(env_fn=[partial(env_fn, cfg=c) for c in collector_env_cfg], cfg=cfg.env.manager)
        evaluator_env = BaseEnvManager(env_fn=[partial(env_fn, cfg=c) for c in evaluator_env_cfg], cfg=cfg.env.manager)

        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

        buffer = DequeBuffer(size=20000)
        policy = SACPolicy(cfg.policy, model=model)

        task.use(basic_evaluator(task, cfg, policy, evaluator_env))
        task.use(basic_collector(task, cfg, policy, collector_env, buffer))
        task.use(basic_learner(task, cfg, policy, buffer))
        task.run(max_step=100000)


if __name__ == "__main__":
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    model = QAC(**cfg.policy.model)
    main(cfg, model)
