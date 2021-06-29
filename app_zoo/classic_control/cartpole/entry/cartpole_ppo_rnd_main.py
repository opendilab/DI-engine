import os
import gym
from tensorboardX import SummaryWriter

from nervex.config import compile_config
from nervex.worker import BaseLearner, SampleCollector, BaseSerialEvaluator, NaiveReplayBuffer
from nervex.envs import BaseEnvManager, NervexEnvWrapper
from nervex.policy import PPOPolicy
from nervex.model import VAC
from nervex.utils import set_pkg_seed, deep_merge_dicts
from nervex.reward_model import RndRewardModel
from app_zoo.classic_control.cartpole.config.cartpole_ppo_rnd_config import cartpole_ppo_rnd_config


def wrapped_cartpole_env():
    return NervexEnvWrapper(gym.make('CartPole-v0'))


def main(cfg, seed=0, max_iterations=int(1e10)):
    cfg = compile_config(
        cfg,
        BaseEnvManager,
        PPOPolicy,
        BaseLearner,
        SampleCollector,
        BaseSerialEvaluator,
        NaiveReplayBuffer,
        reward_model=RndRewardModel,
        save_cfg=True
    )
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    collector_env = BaseEnvManager(env_fn=[wrapped_cartpole_env for _ in range(collector_env_num)], cfg=cfg.env.manager)
    evaluator_env = BaseEnvManager(env_fn=[wrapped_cartpole_env for _ in range(evaluator_env_num)], cfg=cfg.env.manager)

    collector_env.seed(seed)
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    model = VAC(**cfg.policy.model)
    policy = PPOPolicy(cfg.policy, model=model)
    tb_logger = SummaryWriter(os.path.join('./log/', 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger)
    collector = SampleCollector(cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger)
    evaluator = BaseSerialEvaluator(cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger)
    replay_buffer = NaiveReplayBuffer(cfg.policy.other.replay_buffer)
    reward_model = RndRewardModel(cfg.reward_model, policy.collect_mode.get_attribute('device'), tb_logger)

    for _ in range(max_iterations):
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        new_data = collector.collect(train_iter=learner.train_iter)
        assert all([len(c) == 0 for c in collector._traj_buffer.values()])
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        reward_model.collect_data(new_data)
        reward_model.train()
        reward_model.clear_data()
        for i in range(cfg.policy.learn.update_per_collect):
            train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
            reward_model.estimate(train_data)
            if train_data is not None:
                learner.train(train_data, collector.envstep)
        replay_buffer.clear()


if __name__ == "__main__":
    main(cartpole_ppo_rnd_config)
