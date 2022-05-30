from easydict import EasyDict
import pytest
from copy import deepcopy
from typing import List
import os
from functools import partial
from tensorboardX import SummaryWriter

from ding.envs import get_vec_env_setting, create_env_manager
from ding.worker import BaseSerialCommander, create_buffer, create_serial_collector
from ding.config import compile_config
from ding.policy import create_policy
from ding.utils import set_pkg_seed
from ding.entry.utils import random_collect, mark_not_expert, mark_warm_up
from dizoo.classic_control.cartpole.config.cartpole_c51_config import cartpole_c51_config, cartpole_c51_create_config


@pytest.mark.unittest
@pytest.mark.parametrize('collector_type', ['sample', 'episode'])
@pytest.mark.parametrize('transition_with_policy_data', [True, False])
@pytest.mark.parametrize('data_postprocess', [True, False])
def test_random_collect(collector_type, transition_with_policy_data, data_postprocess):

    def mark_not_expert_episode(ori_data: List[List[dict]]) -> List[List[dict]]:
        for i in range(len(ori_data)):
            for j in range(len(ori_data[i])):
                # Set is_expert flag (expert 1, agent 0)
                ori_data[i][j]['is_expert'] = 0
        return ori_data

    def mark_warm_up_episode(ori_data: List[List[dict]]) -> List[List[dict]]:
        for i in range(len(ori_data)):
            for j in range(len(ori_data[i])):
                ori_data[i][j]['warm_up'] = True
        return ori_data

    RANDOM_COLLECT_SIZE = 8
    cfg, create_cfg = deepcopy(cartpole_c51_config), deepcopy(cartpole_c51_create_config)
    cfg.exp_name = "test_cartpole_c51_seed0"
    create_cfg.policy.type = create_cfg.policy.type + '_command'
    cfg.policy.random_collect_size = RANDOM_COLLECT_SIZE
    cfg.policy.transition_with_policy_data = transition_with_policy_data
    if collector_type == 'episode':
        cfg.policy.collect.n_sample = None
        cfg.policy.collect.n_episode = 1
        cfg.policy.collect.n_episode = 1
        cfg.policy.collect.n_episode = 1
        create_cfg.replay_buffer = EasyDict(type=collector_type)
        create_cfg.collector = EasyDict(type=collector_type)
    cfg = compile_config(cfg, seed=0, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)

    # Create main components: env, policy
    env_fn, collector_env_cfg, _ = get_vec_env_setting(cfg.env)
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    collector_env.seed(cfg.seed)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    policy = create_policy(cfg.policy, model=None, enable_field=['learn', 'collect', 'eval', 'command'])

    # Create worker components: collector, replay buffer, commander.
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = EasyDict(learn_info=dict(learner_step=10, priority_info='no_info', learner_done=False))  # Fake Learner
    collector = create_serial_collector(
        cfg.policy.collect.collector,
        env=collector_env,
        policy=policy.collect_mode,
        tb_logger=tb_logger,
        exp_name=cfg.exp_name
    )
    evaluator = None  # Fake Evaluator
    replay_buffer = create_buffer(cfg.policy.other.replay_buffer, tb_logger=tb_logger, exp_name=cfg.exp_name)
    commander = BaseSerialCommander(
        cfg.policy.other.commander, learner, collector, evaluator, replay_buffer, policy.command_mode
    )

    if data_postprocess:
        if collector_type == 'sample':
            postprocess_data_fn = lambda x: mark_warm_up(mark_not_expert(x))
        else:
            postprocess_data_fn = lambda x: mark_warm_up_episode(mark_not_expert_episode(x))
    else:
        postprocess_data_fn = None

    # Accumulate plenty of data at the beginning of training.
    if cfg.policy.get('random_collect_size', 0) > 0:
        random_collect(
            cfg.policy,
            policy,
            collector,
            collector_env,
            commander,
            replay_buffer,
            postprocess_data_fn=postprocess_data_fn
        )
    assert replay_buffer.count() == RANDOM_COLLECT_SIZE
    if data_postprocess:
        if collector_type == 'sample':
            for d in replay_buffer._data[:RANDOM_COLLECT_SIZE]:
                assert d['is_expert'] == 0
                assert d['warm_up'] is True
        else:
            for e in replay_buffer._data[:RANDOM_COLLECT_SIZE]:
                for d in e:
                    assert d['is_expert'] == 0
                    assert d['warm_up'] is True


if __name__ == '__main__':
    test_random_collect()
