import os
from tensorboardX import SummaryWriter

from nervex.worker import BaseLearner, BaseSerialCollector, BaseSerialEvaluator
from nervex.data import BufferManager
from nervex.envs import BaseEnvManager
from nervex.policy import DQNPolicy
from nervex.model import FCDiscreteNet
from nervex.entry.utils import set_pkg_seed
from nervex.rl_utils import epsilon_greedy
from app_zoo.classic_control.cartpole.envs import CartPoleEnv
from app_zoo.classic_control.cartpole.entry.cartpole_dqn_default_config import cartpole_dqn_default_config


def main(cfg, seed=0):
    collector_env_num, evaluator_env_num = cfg.env.env_kwargs.collector_env_num, cfg.env.env_kwargs.evaluator_env_num
    collector_env = BaseEnvManager(env_fn=[lambda: CartPoleEnv() for _ in range(collector_env_num)])
    evaluator_env = BaseEnvManager(env_fn=[lambda: CartPoleEnv() for _ in range(evaluator_env_num)])

    collector_env.seed(seed)
    evaluator_env.seed(seed)
    set_pkg_seed(seed, use_cuda=cfg.policy.use_cuda)

    model = FCDiscreteNet(**cfg.policy.model)
    policy = DQNPolicy(cfg.policy, model=model)
    tb_logger = SummaryWriter(os.path.join('./log/', 'serial'))
    learner = BaseLearner(cfg.learner, policy.learn_mode, tb_logger)
    collector = BaseSerialCollector(cfg.collector, collector_env, policy.collect_mode, tb_logger)
    evaluator = BaseSerialEvaluator(cfg.evaluator, evaluator_env, policy.eval_mode, tb_logger)
    replay_buffer = BufferManager(cfg.replay_buffer, tb_logger)

    eps_cfg = cfg.policy.other.eps
    epsilon_greedy_fn = epsilon_greedy(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

    while True:
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        eps = epsilon_greedy_fn(learner.train_iter)
        new_data = collector.collect_data(learner.train_iter, policy_kwargs={'eps': eps})
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        for i in range(cfg.policy.learn.train_iteration):
            train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
            if train_data is not None:
                learner.train(train_data, collector.envstep)
                if cfg.policy.get('use_priority', False):
                    replay_buffer.update(learner.priority_info)


if __name__ == "__main__":
    main(cartpole_dqn_default_config, seed=0)
