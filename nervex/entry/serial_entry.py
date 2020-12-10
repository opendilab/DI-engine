import sys
import copy
import time
import argparse
import numpy as np
import torch

from nervex.worker import SubprocessEnvManager, BaseLearner, BaseSerialActor, BaseSerialEvaluator
from nervex.utils import read_config
from nervex.data import ReplayBuffer
from nervex.policy import create_policy
from nervex.envs import get_vec_env_setting


def main(args):
    cfg = read_config(args.config_path)
    env_fn, actor_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    actor_env = SubprocessEnvManager(env_fn=env_fn, env_cfg=actor_env_cfg, env_num=len(actor_env_cfg))
    evaluator_env = SubprocessEnvManager(env_fn, env_cfg=evaluator_env_cfg, env_num=len(evaluator_env_cfg))
    # seed
    actor_env.seed(args.seed)
    evaluator_env.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    # create component
    policy = create_policy(cfg.policy)
    replay_buffer = ReplayBuffer(cfg.replay_buffer)
    learner = BaseLearner(cfg)
    actor = BaseSerialActor(cfg.actor)
    evaluator = BaseSerialEvaluator(cfg.evaluator)
    actor.env = actor_env
    evaluator.env = evaluator_env
    learner.policy = policy.learn_mode
    actor.policy = policy.collect_mode
    evaluator.policy = policy.eval_mode
    learner.launch()
    # main loop
    iter_count = 0
    while True:
        new_data = actor.generate_data()
        replay_buffer.push_data(new_data)
        train_data = replay_buffer.sample(cfg.policy.learn.batch_size)
        learner.train(train_data)
        if (iter_count + 1) % cfg.evaluator.eval_freq == 0 and evaluator.eval():
            learner.save_checkpoint()
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
