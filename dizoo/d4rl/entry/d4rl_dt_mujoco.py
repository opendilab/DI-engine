import gym
import torch
import numpy as np
from ditk import logging
from ding.model.template.decision_transformer import DecisionTransformer
from ding.policy import DTPolicy
from ding.envs import BaseEnvManagerV2
from ding.envs.env_wrappers.env_wrappers import AllinObsWrapper
from ding.data import create_dataset
from ding.config import compile_config
from ding.framework import task, ding_init
from ding.framework.context import OfflineRLContext
from ding.framework.middleware import interaction_evaluator, trainer, CkptSaver, offline_data_fetcher, offline_logger, termination_checker
from ding.utils import set_pkg_seed
from dizoo.d4rl.envs import D4RLEnv
from dizoo.d4rl.config.hopper_medium_dt_config import main_config, create_config


def main():
    # If you don't have offline data, you need to prepare if first and set the data_path in config
    # For demostration, we also can train a RL policy (e.g. SAC) and collect some data
    logging.getLogger().setLevel(logging.INFO)
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    ding_init(cfg)
    with task.start(async_mode=False, ctx=OfflineRLContext()):
        evaluator_env = BaseEnvManagerV2(
            env_fn=[lambda: AllinObsWrapper(D4RLEnv(cfg.env)) for _ in range(cfg.env.evaluator_env_num)],
            cfg=cfg.env.manager
        )

        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

        dataset = create_dataset(cfg)
        # env_data_stats = dataset.get_d4rl_dataset_stats(cfg.policy.dataset_name)
        env_data_stats = dataset.get_state_stats()
        cfg.policy.state_mean, cfg.policy.state_std = np.array(env_data_stats['state_mean']
                                                               ), np.array(env_data_stats['state_std'])
        model = DecisionTransformer(**cfg.policy.model)
        policy = DTPolicy(cfg.policy, model=model)
        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(offline_data_fetcher(cfg, dataset))
        task.use(trainer(cfg, policy.learn_mode))
        task.use(termination_checker(max_train_iter=1e5))
        task.use(CkptSaver(policy, cfg.exp_name, train_freq=100))
        task.use(offline_logger())
        task.run()


if __name__ == "__main__":
    main()
