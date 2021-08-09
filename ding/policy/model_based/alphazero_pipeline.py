import os
import random
from collections import deque

import numpy as np

from ding.config.config import read_config_yaml
from ding.policy.model_based.alphazero import AlphaZeroPolicy
from ding.policy.model_based.alphazero_collector import AlphazeroCollector


def serial_alphazero_pipeline(
        cfg_path,
) -> 'Policy':  # noqa
    """
    Overview:
        Serial pipeline entry.
    Arguments:
        - input_cfg (:obj:`Union[str, Tuple[dict, dict]]`): Config in dict type. \
            ``str`` type means config file path. \
            ``Tuple[dict, dict]`` type means [user_config, create_cfg].
        - seed (:obj:`int`): Random seed.
        - env_setting (:obj:`Optional[List[Any]]`): A list with 3 elements: \
            ``BaseEnv`` subclass, collector env config, and evaluator env config.
        - model (:obj:`Optional[torch.nn.Module]`): Instance of torch.nn.Module.
        - max_iterations (:obj:`Optional[torch.nn.Module]`): Learner's max iteration. Pipeline will stop \
            when reaching this iteration.
    Returns:
        - policy (:obj:`Policy`): Converged policy.
    """
    cfg = read_config_yaml(cfg_path)
    # Create main components: env, policy
    agent = AlphaZeroPolicy(cfg)
    collector = AlphazeroCollector(cfg, agent=agent)
    evaluator = AlphazeroCollector(cfg, agent=agent)
    replay_buffer = deque(maxlen=cfg.replay_buffer.buffer_size)
    batch_size = cfg.learner.batch_size
    for iter_count in range(1500):
        for idx in range(cfg.collector.collect_n_episodes):
            winner, play_data = collector.start_self_play()
            # augment the data
            play_data = collector.get_equi_data(play_data)
            replay_buffer.extend(play_data)
        if len(replay_buffer) < batch_size:
            continue
        else:
            mini_batch = random.sample(replay_buffer, batch_size)
            agent.update_policy(mini_batch, cfg.learner.update_per_collect)
        if iter_count % 100 == 0:
            winner_list_first, winner_list_next = evaluator.start_play_with_expert(test_episode_num=10)
            print(np.mean(winner_list_first), np.mean(winner_list_next))

    return True


if __name__ == '__main__':
    cfg_path = os.path.join(os.getcwd(), 'alphazero_config.yaml')
    serial_alphazero_pipeline(cfg_path)
