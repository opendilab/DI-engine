import sys
import copy
import time
import argparse
import numpy as np
import torch

from nervex.worker import SubprocessEnvManager, BaseLearner, BaseSerialActor, BaseSerialEvaluator
from nervex.utils import read_config
from nervex.data import ReplayBuffer
from nervex.model import FCDiscreteNet
from nervex.policy import DQNPolicy
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
    e_info = actor_env.env_info()
    model = FCDiscreteNet(e_info.obs_space.shape, e_info.act_space.shape, cfg.model.embedding_dim)
    policy = DQNPolicy(cfg.policy, model)
    replay_buffer = ReplayBuffer(cfg.replay_buffer)
    learner = BaseLearner(cfg)
    actor = BaseSerialActor(cfg)
    evaluator = BaseSerialEvaluator(cfg)
    learner.policy = policy.learn
    actor.policy = policy.collect
    evaluator.policy = policy.eval
    # main loop
    while True:
        new_data = actor.generate_data()
        replay_buffer.push_data(new_data)
        train_data = replay_buffer.sample(cfg.learner.data.batch_size)
        learner.train(train_data)
        if evaluator.eval():
            break
        if cfg.learner.on_policy:
            replay_buffer.clear()

    # close
    actor_env.close()
    evaluator_env.close()
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
