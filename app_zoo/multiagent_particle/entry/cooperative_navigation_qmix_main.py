import os
import gym
from tensorboardX import SummaryWriter

from nervex.worker import BaseLearner, BaseSerialCollector, BaseSerialEvaluator
from nervex.data import BufferManager
from nervex.envs import SyncSubprocessEnvManager
from app_zoo.multiagent_particle.envs import CooperativeNavigation
from nervex.policy import QMIXPolicy
from nervex.model import QMix
from nervex.entry.utils import set_pkg_seed
from nervex.rl_utils import get_epsilon_greedy_fn
from app_zoo.multiagent_particle.config import cooperative_navigation_qmix_default_config


def wrapped_env(cfg):
        return lambda : CooperativeNavigation(cfg=cfg)


def main(cfg, seed=0):
    collector_env_num, evaluator_env_num = cfg.env.env_kwargs.collector_env_num, cfg.env.env_kwargs.evaluator_env_num
    collector_env = SyncSubprocessEnvManager(
        env_fn=[wrapped_env(cfg.env) for _ in range(collector_env_num)])
    evaluator_env = SyncSubprocessEnvManager(
        env_fn=[wrapped_env(cfg.env) for _ in range(evaluator_env_num)])
    
    collector_env.seed(seed)
    evaluator_env.seed(seed)
    set_pkg_seed(seed, use_cuda=cfg.policy.use_cuda)

    model = QMix(**cfg.policy.model)
    policy = QMIXPolicy(cfg.policy, model=model)
    tb_logger = SummaryWriter(os.path.join('./log/', 'serial'))
    learner = BaseLearner(cfg.learner, policy.learn_mode, tb_logger)
    collector = BaseSerialCollector(
        cfg.collector, collector_env, policy.collect_mode, tb_logger)
    evaluator = BaseSerialEvaluator(
        cfg.evaluator, evaluator_env, policy.eval_mode, tb_logger)
    replay_buffer = BufferManager(cfg.replay_buffer, tb_logger)

    eps_cfg = cfg.policy.other.eps
    epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

    while True:
        eps = epsilon_greedy(learner.train_iter)
        new_data = collector.collect_data(learner.train_iter, policy_kwargs={'eps': eps})
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        for i in range(cfg.policy.learn.train_iteration):
            train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
            if train_data is not None:
                learner.train(train_data, collector.envstep)
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break


if __name__ == "__main__":
    main(cooperative_navigation_qmix_default_config, seed=0)
