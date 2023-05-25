import gym
from ditk import logging
from ding.model import QACEnsemble
from ding.policy import EDACPolicy
from ding.envs import DingEnvWrapper, BaseEnvManagerV2
from ding.data import create_dataset
from ding.config import compile_config
from ding.framework import task, ding_init
from ding.framework.context import OfflineRLContext
from ding.framework.middleware import interaction_evaluator, trainer, CkptSaver, offline_data_fetcher, offline_logger
from ding.utils import set_pkg_seed
from dizoo.d4rl.envs import D4RLEnv
from dizoo.d4rl.config.halfcheetah_medium_edac_config import main_config, create_config


def main():
    # If you don't have offline data, you need to prepare if first and set the data_path in config
    # For demostration, we also can train a RL policy (e.g. SAC) and collect some data
    logging.getLogger().setLevel(logging.INFO)
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    ding_init(cfg)
    with task.start(async_mode=False, ctx=OfflineRLContext()):
        evaluator_env = BaseEnvManagerV2(
            env_fn=[lambda: D4RLEnv(cfg.env) for _ in range(cfg.env.evaluator_env_num)], cfg=cfg.env.manager
        )

        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

        dataset = create_dataset(cfg)
        model = QACEnsemble(**cfg.policy.model)
        policy = EDACPolicy(cfg.policy, model=model)

        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(offline_data_fetcher(cfg, dataset))
        task.use(trainer(cfg, policy.learn_mode))
        task.use(CkptSaver(policy, cfg.exp_name, train_freq=1e4))
        task.use(offline_logger())
        task.run()


if __name__ == "__main__":
    main()
