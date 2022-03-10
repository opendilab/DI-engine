import os
import gym
from tensorboardX import SummaryWriter
from easydict import EasyDict

from ding.config import compile_config
from ding.worker import BaseLearner, SampleSerialCollector, InteractionSerialEvaluator, NaiveReplayBuffer
from ding.envs import BaseEnvManager, DingEnvWrapper
from ding.policy import PPOPolicy
from ding.model import VAC
from ding.utils import set_pkg_seed
from dizoo.box2d.car_racing.envs import CarRacingEnv
from dizoo.box2d.car_racing.config.car_racing_ppo_config import car_racing_ppo_config


def main(cfg, seed=0, max_iterations=int(1e10)):
    from easydict import EasyDict
    cfg = EasyDict(cfg)
    cfg = compile_config(
        cfg,
        BaseEnvManager,
        PPOPolicy,
        BaseLearner,
        SampleSerialCollector,
        InteractionSerialEvaluator,
        NaiveReplayBuffer,
        save_cfg=True
    )
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    collector_env = BaseEnvManager(
        env_fn=[lambda: CarRacingEnv(cfg.env) for _ in range(collector_env_num)], cfg=cfg.env.manager
    )
    evaluator_env = BaseEnvManager(
        env_fn=[lambda: CarRacingEnv(cfg.env) for _ in range(evaluator_env_num)], cfg=cfg.env.manager
    )

    collector_env.seed(seed)
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    model = VAC(**cfg.policy.model)
    policy = PPOPolicy(cfg.policy, model=model)
    tb_logger = SummaryWriter(os.path.join('../../../../../Opendi/2022-3-10/tb_ppo_car_racing/log/', 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger)
    collector = SampleSerialCollector(cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger)
    evaluator = InteractionSerialEvaluator(cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger)

    for _ in range(max_iterations):
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        new_data = collector.collect(train_iter=learner.train_iter)
        learner.train(new_data, collector.envstep)


if __name__ == "__main__":
    main(car_racing_ppo_config)
