import sys
import copy
import time
import argparse
import numpy as np
import torch

from nervex.worker import BaseLearner, BaseSerialActor, BaseSerialEvaluator, BaseSerialCommand
from nervex.worker import BaseEnvManager, SubprocessEnvManager
from nervex.utils import read_config
from nervex.data import ReplayBuffer
from nervex.policy import create_policy
from nervex.envs import get_vec_env_setting


def main(args):
    cfg = read_config(args.config_path)
    env_fn, actor_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    env_manager_type = BaseEnvManager if cfg.env.env_manager_type == 'base' else SubprocessEnvManager
    actor_env = env_manager_type(env_fn=env_fn, env_cfg=actor_env_cfg, env_num=len(actor_env_cfg))
    evaluator_env = env_manager_type(env_fn, env_cfg=evaluator_env_cfg, env_num=len(evaluator_env_cfg))
    # seed
    actor_env.seed(args.seed)
    evaluator_env.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    # create component
    policy = create_policy(cfg.policy)
    learner = BaseLearner(cfg)
    actor = BaseSerialActor(cfg.actor)
    evaluator = BaseSerialEvaluator(cfg.evaluator)
    replay_buffer = ReplayBuffer(cfg.replay_buffer)
    command = BaseSerialCommand(cfg.command, learner, actor, evaluator, replay_buffer)

    actor.env = actor_env
    evaluator.env = evaluator_env
    learner.policy = policy.learn_mode
    actor.policy = policy.collect_mode
    evaluator.policy = policy.eval_mode
    command.policy = policy.command_mode
    learner.launch()
    # main loop
    iter_count = 0
    n_step = cfg.policy.learn.batch_size if iter_count == 0 else cfg.actor.n_step
    while True:
        command.step()
        new_data = actor.generate_data(n_step=n_step)
        replay_buffer.push_data(new_data)
        train_data = replay_buffer.sample(cfg.policy.learn.batch_size)
        learner.train(train_data)
        if (iter_count + 1) % cfg.evaluator.eval_freq == 0 and evaluator.eval():
            learner.save_checkpoint()
            print("Your RL agent is converged, you can refer to 'log/evaluator.txt' for details")
            break
        if cfg.policy.on_policy:
            replay_buffer.clear()
        iter_count += 1

    # close
    replay_buffer.close()
    learner.close()
    actor.close()
    evaluator.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default=None)
    parser.add_argument('--seed', default=0)
    args = parser.parse_known_args()[0]
    main(args)
