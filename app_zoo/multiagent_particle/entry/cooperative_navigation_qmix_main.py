import os
import gym
from tensorboardX import SummaryWriter
from easydict import EasyDict

from nervex.config import compile_config
from nervex.worker import BaseLearner, BaseSerialCollector, BaseSerialEvaluator, PrioritizedReplayBuffer
from nervex.envs import SyncSubprocessEnvManager
from nervex.policy import QMIXPolicy
from nervex.model import QMix
from nervex.utils import set_pkg_seed, deep_merge_dicts
from nervex.rl_utils import get_epsilon_greedy_fn
from app_zoo.multiagent_particle.envs import CooperativeNavigation
from app_zoo.multiagent_particle.config.cooperative_navigation_qmix_config import cooperative_navigation_qmix_config


def main(cfg, seed=0):

    def wrapped_env():
        return CooperativeNavigation(cfg=cfg.env)

    cfg = compile_config(
        cfg,
        SyncSubprocessEnvManager,
        QMIXPolicy,
        BaseLearner,
        BaseSerialCollector,
        BaseSerialEvaluator,
        PrioritizedReplayBuffer,
        save_cfg=True
    )
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    collector_env = SyncSubprocessEnvManager(
        env_fn=[wrapped_env for _ in range(collector_env_num)], cfg=cfg.env.manager
    )
    evaluator_env = SyncSubprocessEnvManager(
        env_fn=[wrapped_env for _ in range(evaluator_env_num)], cfg=cfg.env.manager
    )

    collector_env.seed(seed)
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    model = QMix(**cfg.policy.model)
    policy = QMIXPolicy(cfg.policy, model=model)
    tb_logger = SummaryWriter(os.path.join('./log/', 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger)
    collector = BaseSerialCollector(cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger)
    evaluator = BaseSerialEvaluator(cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger)
    replay_buffer = PrioritizedReplayBuffer('default_buffer', cfg.policy.other.replay_buffer, tb_logger)

    eps_cfg = cfg.policy.other.eps
    epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

    while True:
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        eps = epsilon_greedy(collector.envstep)
        new_data = collector.collect_data(learner.train_iter, policy_kwargs={'eps': eps})
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        for i in range(cfg.policy.learn.update_per_collect):
            train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
            if train_data is not None:
                learner.train(train_data, collector.envstep)


if __name__ == "__main__":
    main(cooperative_navigation_qmix_config)
