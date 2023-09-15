"""
# Example of PPO pipeline

Use the pipeline on a single process:

> python3 -u ding/example/ppo.py

Use the pipeline on multiple processes:

We surpose there are N processes (workers) = 1 learner + 1 evaluator + (N-2) collectors

## First Example —— Execute on one machine with multi processes.

Execute 4 processes with 1 learner + 1 evaluator + 2 collectors
Remember to keep them connected by mesh to ensure that they can exchange information with each other.

> ditask --package . --main ding.example.ppo.main --parallel-workers 4 --topology mesh
"""
import gym
from ditk import logging
from ding.model import VAC
from ding.policy import PPOPolicy
from ding.envs import DingEnvWrapper, BaseEnvManagerV2
from ding.data import DequeBuffer
from ding.config import compile_config
from ding.framework import task, ding_init
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import multistep_trainer, StepCollector, interaction_evaluator, CkptSaver, \
    gae_estimator, online_logger, ContextExchanger, ModelExchanger
from ding.utils import set_pkg_seed
from dizoo.classic_control.cartpole.config.cartpole_ppo_config import main_config, create_config


def main():
    logging.getLogger().setLevel(logging.INFO)
    cfg = compile_config(main_config, create_cfg=create_config, auto=True, save_cfg=task.router.node_id == 0)
    ding_init(cfg)
    with task.start(async_mode=False, ctx=OnlineRLContext()):
        collector_env = BaseEnvManagerV2(
            env_fn=[lambda: DingEnvWrapper(gym.make("CartPole-v0")) for _ in range(cfg.env.collector_env_num)],
            cfg=cfg.env.manager
        )
        evaluator_env = BaseEnvManagerV2(
            env_fn=[lambda: DingEnvWrapper(gym.make("CartPole-v0")) for _ in range(cfg.env.evaluator_env_num)],
            cfg=cfg.env.manager
        )

        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

        model = VAC(**cfg.policy.model)
        policy = PPOPolicy(cfg.policy, model=model)

        # Consider the case with multiple processes
        if task.router.is_active:
            # You can use labels to distinguish between workers with different roles,
            # here we use node_id to distinguish.
            if task.router.node_id == 0:
                task.add_role(task.role.LEARNER)
            elif task.router.node_id == 1:
                task.add_role(task.role.EVALUATOR)
            else:
                task.add_role(task.role.COLLECTOR)

            # Sync their context and model between each worker.
            task.use(ContextExchanger(skip_n_iter=1))
            task.use(ModelExchanger(model))

        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(StepCollector(cfg, policy.collect_mode, collector_env))
        task.use(gae_estimator(cfg, policy.collect_mode))
        task.use(multistep_trainer(policy.learn_mode, log_freq=50))
        task.use(CkptSaver(policy, cfg.exp_name, train_freq=100))
        task.use(online_logger(train_show_freq=3))
        task.run()


if __name__ == "__main__":
    main()
