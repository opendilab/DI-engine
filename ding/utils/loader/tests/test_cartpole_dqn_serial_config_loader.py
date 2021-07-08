import math

import pytest

from dizoo.classic_control.cartpole.config import cartpole_dqn_config, cartpole_dqn_create_config
from ding.utils.loader import dict_, is_type, to_type, collection, interval, is_positive, mcmp, enum, item, raw, \
    check_only
from ding.utils import pretty_print


@pytest.mark.unittest
def test_main_config():
    element_loader = dict_(
        env=item('env') >> dict_(
            collector_env_num=item('collector_env_num') >> is_type(int) >> interval(1, 32),
            evaluator_env_num=item('evaluator_env_num') >> is_type(int) >> interval(1, 32),
        ),
        policy=item('policy') >> dict_(
            type=item('type') | raw('dqn') >> is_type(str),
            cuda=item('cuda') >> is_type(bool),
            on_policy=item('on_policy') | raw(False) >> is_type(bool),
            priority=item('priority') | raw(False) >> is_type(bool),
            model=item('model') >> dict_(
                obs_dim=item('obs_shape') >> (is_type(int) | collection(int)),
                action_dim=item('action_shape') >> (is_type(int) | collection(int)),
                hidden_size_list=item('encoder_hidden_size_list') >> is_type(list),
                dueling=item('dueling') >> is_type(bool),
            ),
            learn=item('learn') >> dict_(
                multi_gpu=item('multi_gpu') | raw(False) >> is_type(bool),
                update_per_collect=item('update_per_collect') | raw(1) >> (is_type(int) & interval(1, 500)),
                batch_size=item('batch_size') | raw(64) >> (is_type(int) & interval(1, 128)),
                learning_rate=item('learning_rate') | raw(0.001) >> interval(0.0001, 0.01),
                target_update_freq=item('target_update_freq') | raw(200) >> (is_type(int) & interval(100, 2000)),
                discount_factor=item('discount_factor') | raw(0.99) >> interval(0.9, 1.0),
                nstep=item('nstep') | raw(1) >> (is_type(int) & interval(1, 10)),
                ignore_done=item('ignore_done') | raw(False) >> is_type(bool),
            ),
            collect=item('collect') >> dict_(
                n_sample=item('n_sample') | raw(20) >> is_type(int) >> interval(8, 128),
                n_episode=item('n_episode') | raw(10) >> is_type(int) >> interval(2, 10),
                unroll_len=item('unroll_len') | raw(1) >> is_type(int) >> interval(1, 200),
                nstep=item('nstep') | raw(1) >> (is_type(int) & interval(1, 10)),
            ),
            other=item('other') >> dict_(
                eps=item('eps') >> dict_(
                    type=item('type') >> enum('linear', 'exp'),
                    start=item('start') >> interval(0.0, 1.0, left_ok=False),
                    end=item('end') >> interval(0.0, 1.0, right_ok=False),
                    decay=item('decay') >> (is_type(int) | (is_type(float) >> to_type(int))) >> is_positive(),
                ),
                replay_buffer=item('replay_buffer') >>
                dict_(replay_buffer_size=item('replay_buffer_size') >> is_type(int) >> interval(1, math.inf), ),
            ),
        ),
    )
    learn_nstep = item('policy') >> item('learn') >> item('nstep')
    collect_nstep = item('policy') >> item('collect') >> item('nstep')
    relation_loader = check_only(
        dict_(
            nstep_check=mcmp(learn_nstep, "==", collect_nstep),
            eps_check=item('policy') >> item('other') >> item('eps') >> mcmp(item('start'), ">=", item('end')),
        )
    )
    cartpole_dqn_main_loader = element_loader >> relation_loader

    output = cartpole_dqn_main_loader(cartpole_dqn_config)
    pretty_print(output, direct_print=True)


@pytest.mark.unittest
def test_create_config():
    element_loader = dict_(
        env=item('env') >> dict_(
            import_names=item('import_names') >> collection(str),
            type=item('type') >> is_type(str),
        ),
        env_manager=item('env_manager') >> dict_(
            type=item('type') >> enum('base', 'subprocess', 'async_subprocess'),
            shared_memory=item('shared_memory') | raw(True) >> is_type(bool),
        ),
        policy=item('policy') >> dict_(type=item('type') >> is_type(str), ),
    )
    cartpole_dqn_create_loader = element_loader

    output = cartpole_dqn_create_loader(cartpole_dqn_create_config)
    pretty_print(output, direct_print=True)
