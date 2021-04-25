import os
import gym
from tensorboardX import SummaryWriter

from nervex.worker import BaseLearner, BaseSerialCollector, BaseSerialEvaluator
from nervex.data import BufferManager
from nervex.envs import BaseEnvManager, NervexEnvWrapper
from nervex.policy import DDPGPolicy
from nervex.model import QAC
from nervex.entry.utils import set_pkg_seed
from app_zoo.classic_control.pendulum.config import pendulum_td3_default_config


def main(cfg, seed=0):
    
    def wrapped_pendulum_env():
        return NervexEnvWrapper(gym.make('Pendulum-v0'), cfg=cfg.env.wrapper)
    
    collector_env_num, evaluator_env_num = cfg.env.env_kwargs.collector_env_num, cfg.env.env_kwargs.evaluator_env_num
    collector_env = BaseEnvManager(
        env_fn=[wrapped_pendulum_env for _ in range(collector_env_num)])
    evaluator_env = BaseEnvManager(
        env_fn=[wrapped_pendulum_env for _ in range(evaluator_env_num)])

    collector_env.seed(seed)
    evaluator_env.seed(seed)
    set_pkg_seed(seed, use_cuda=cfg.policy.use_cuda)

    model = QAC(**cfg.policy.model)
    policy = DDPGPolicy(cfg.policy, model=model)
    tb_logger = SummaryWriter(os.path.join('./log/', 'serial'))
    learner = BaseLearner(cfg.learner, policy.learn_mode, tb_logger)
    collector = BaseSerialCollector(
        cfg.collector, collector_env, policy.collect_mode, tb_logger)
    evaluator = BaseSerialEvaluator(
        cfg.evaluator, evaluator_env, policy.eval_mode, tb_logger)
    replay_buffer = BufferManager(cfg.replay_buffer, tb_logger)

    while True:
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        new_data = collector.collect_data(learner.train_iter)
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        for i in range(cfg.policy.learn.train_iteration):
            train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
            if train_data is not None:
                learner.train(train_data, collector.envstep)
                replay_buffer.update(learner.priority_info)


if __name__ == "__main__":
    main(pendulum_td3_default_config, seed=0)
