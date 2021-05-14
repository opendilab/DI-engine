import os
import gym
from tensorboardX import SummaryWriter
from easydict import EasyDict

from nervex.config import compile_config
from nervex.worker import BaseLearner, BaseSerialCollector, BaseSerialEvaluator
from nervex.data import BufferManager
from nervex.envs import BaseEnvManager, NervexEnvWrapper
from nervex.policy import DQNPolicy
from nervex.model import FCDiscreteNet
from nervex.utils import set_pkg_seed
from nervex.rl_utils import get_epsilon_greedy_fn
from app_zoo.classic_control.cartpole.envs import CartPoleEnv
from app_zoo.classic_control.cartpole.config.cartpole_dqn_config import cartpole_dqn_config


# Get nerveX form env class
def wrapped_cartpole_env():
    return NervexEnvWrapper(gym.make('CartPole-v0'))


def main(cfg, seed=0):
    cfg = compile_config(
        cfg,
        CartPoleEnv,
        BaseEnvManager,
        DQNPolicy,
        BaseLearner,
        BaseSerialCollector,
        BaseSerialEvaluator,
        BufferManager,
        save_cfg=True
    )
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    # collector_env = BaseEnvManager(env_fn=[wrapped_cartpole_env for _ in range(collector_env_num)], cfg=cfg.env.manager)
    # evaluator_env = BaseEnvManager(env_fn=[wrapped_cartpole_env for _ in range(evaluator_env_num)], cfg=cfg.env.manager)
    collector_env = BaseEnvManager(
        env_fn=[lambda: CartPoleEnv({}) for _ in range(collector_env_num)], cfg=cfg.env.manager
    )
    evaluator_env = BaseEnvManager(
        env_fn=[lambda: CartPoleEnv({}) for _ in range(evaluator_env_num)], cfg=cfg.env.manager
    )

    # Set random seed for all package and instance
    collector_env.seed(seed)
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    # Set up RL Policy
    model = FCDiscreteNet(**cfg.policy.model)
    policy = DQNPolicy(cfg.policy, model=model)

    # Set up collection, training and evaluation utilities
    tb_logger = SummaryWriter(os.path.join('./log/', 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger)
    collector = BaseSerialCollector(cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger)
    evaluator = BaseSerialEvaluator(cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger)
    replay_buffer = BufferManager(cfg.policy.other.replay_buffer, tb_logger)

    # Set up other modules, etc. epsilon greedy
    eps_cfg = cfg.policy.other.eps
    epsilon_greedy = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

    # Training & Evaluation loop
    while True:
        # Evaluating at the beginning and with specific frequency
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        # Update other modules
        eps = epsilon_greedy(collector.envstep)
        # Sampling data from environments
        new_data = collector.collect_data(learner.train_iter, policy_kwargs={'eps': eps})
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        # Training
        for i in range(cfg.policy.learn.update_per_collect):
            train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
            if train_data is not None:
                learner.train(train_data, collector.envstep)


if __name__ == "__main__":
    main(cartpole_dqn_config)
