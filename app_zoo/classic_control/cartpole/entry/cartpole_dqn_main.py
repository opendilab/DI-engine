import os
from tensorboardX import SummaryWriter

from nervex.worker import BaseLearner, BaseSerialActor, BaseSerialEvaluator, BaseSerialCommander
from nervex.data import BufferManager
from nervex.envs import BaseEnvManager
from nervex.policy import DQNPolicy
from nervex.model import FCDiscreteNet
from nervex.entry.utils import set_pkg_seed
from app_zoo.classic_control.cartpole.envs import CartPoleEnv
from app_zoo.classic_control.cartpole.entry.cartpole_dqn_default_config import cartpole_dqn_default_config


def main(cfg, seed=0):
    actor_env_num, evaluator_env_num = cfg.env.env_kwargs.actor_env_num, cfg.env.env_kwargs.evaluator_env_num
    actor_env = BaseEnvManager(env_fn=[lambda: CartPoleEnv() for _ in range(actor_env_num)])
    evaluator_env = BaseEnvManager(env_fn=[lambda: CartPoleEnv() for _ in range(evaluator_env_num)])

    actor_env.seed(seed)
    evaluator_env.seed(seed)
    set_pkg_seed(seed, use_cuda=cfg.policy.use_cuda)

    model = FCDiscreteNet(**cfg.policy.model)
    policy = DQNPolicy(cfg.policy, model=model)
    tb_logger = SummaryWriter(os.path.join('./log/', 'serial'))
    learner = BaseLearner(cfg.learner, policy.learn_mode, tb_logger)
    actor = BaseSerialActor(cfg.actor, actor_env, policy.collect_mode, tb_logger)
    evaluator = BaseSerialEvaluator(cfg.evaluator, evaluator_env, policy.eval_mode, tb_logger)
    replay_buffer = BufferManager(cfg.replay_buffer, tb_logger)
    commander = BaseSerialCommander(cfg.commander, learner, actor, evaluator, replay_buffer, policy.command_mode)

    while True:
        commander.step()
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, actor.envstep)
            if stop:
                break
        new_data = actor.generate_data(learner.train_iter)
        replay_buffer.push(new_data, cur_actor_envstep=actor.envstep)
        for i in range(cfg.policy.learn.train_iteration):
            train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
            if train_data is not None:
                learner.train(train_data, actor.envstep)
                if cfg.policy.get('use_priority', False):
                    replay_buffer.update(learner.priority_info)


if __name__ == "__main__":
    main(cartpole_dqn_default_config, seed=0)
