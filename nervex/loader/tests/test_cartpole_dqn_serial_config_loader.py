import math
import pytest

from nervex.config.loader.dict import dict_
from nervex.config.loader.types import is_type, to_type
from nervex.config.loader.collection import collection
from nervex.config.loader.number import interval, is_positive, mcmp
from nervex.config.loader.string import enum
from nervex.config.loader.mapping import item
from nervex.config.loader.utils import keep, optional, raw
from nervex.config.loader.norm import norm
from nervex.config.loader.base import Loader
from nervex.utils import pretty_print
from app_zoo.classic_control.cartpole.entry.cartpole_dqn_default_config import cartpole_dqn_default_config


@pytest.mark.unittest
def test_real_loader():
    element_loader = dict_(
        env=item('env') >> dict_(
            env_manager_type=item('env_manager_type') >> enum('base', 'subprocess'),
            import_names=item('import_names') >> collection(str),
            env_type=item('env_type') >> is_type(str),
            actor_env_num=item('actor_env_num') >> interval(1, 32),
            evaluator_env_num=item('evaluator_env_num') >> interval(1, 32),
            manager=item('manager') >> dict_(
                shared_memory=item('shared_memory') >> is_type(bool),
                context=item('context') >> enum('fork', 'spawn', 'forkserver') | raw('fork')
            ),
        ),
        policy=item('policy') >> dict_(
            use_cuda=item('use_cuda') >> is_type(bool),
            policy_type=item('policy_type') >> is_type(str),
            import_names=item('import_names') >> collection(str),
            on_policy=item('on_policy') >> is_type(bool),
            model=item('model') >> dict_(
                obs_dim=item('obs_dim') >> (is_type(int) | collection(int)),
                action_dim=item('action_dim') >> (is_type(int) | collection(int))
            ),
            learn=item('learn') >> dict_(
                batch_size=item('batch_size') >> (is_type(int) & interval(1, 128)),
                learning_rate=item('learning_rate') >> interval(0.0001, 0.01),
                weight_decay=item('weight_decay') >> interval(0.0, 0.001),
                algo=item('algo') >> dict_(
                    target_update_freq=item('target_update_freq') >> (is_type(int) & interval(100, 2000)),
                    discount_factor=item('discount_factor') >> interval(0.9, 1.0),
                    nstep=item('nstep') >> (is_type(int) & interval(1, 10)),
                ),
            ),
            collect=item('collect') >> dict_(
                traj_len=item('traj_len') >> (interval(1, 200) | to_type(float)),
                unroll_len=item('unroll_len') >> interval(1, 200),
                algo=item('algo') >> dict_(nstep=item('nstep') >> (is_type(int) & interval(1, 10))),
            ),
            command=item('command') >> dict_(
                eps=item('eps') >> dict_(
                    type=item('type') >> enum('linear', 'exp'),
                    start=item('start') >> interval(0.0, 1.0, left_ok=False),
                    end=item('end') >> interval(0.0, 1.0, right_ok=False),
                    decay=item('decay') >> (is_type(int) | (is_type(float) >> to_type(int))) >> is_positive(),
                )
            ),
        ),
        replay_buffer=item('replay_buffer') >> dict_(
            buffer_name=item('buffer_name') >> collection(str),
            agent=item('agent') >> dict_(
                meta_maxlen=item('meta_maxlen') >> interval(1, math.inf),
                max_reuse=item('max_reuse') >> interval(1, math.inf),
                min_sample_ratio=item('min_sample_ratio') >> interval(1.0, 10.0)
            ),
        ),
        learner=item('learner') >> dict_(load_path=item('load_path') >> is_type(str)),
        commander=item('commander') | raw({}),
        actor=item('actor') >> dict_(
            n_sample=item('n_sample') >> interval(8, 128),
            traj_len=item('traj_len') >> (interval(1, 200) | to_type(float)),
            traj_print_freq=item('traj_print_freq') >> interval(1, 1000),
            collect_print_freq=item('traj_print_freq') >> interval(1, 1000),
        ),
        evaluator=item('evaluator') >> dict_(
            n_episode=item('n_episode') >> interval(2, 10),
            eval_freq=item('eval_freq') >> interval(100, 500),
        ),
    )
    error_loader = Loader((lambda x: x > 0, lambda x: ValueError('value is {x}'.format(x=x))))
    relation_loader = dict_(
        nstep_check=mcmp(
            item('policy') >> item('learn') >> item('algo') >> item('nstep'), "==",
            item('policy') >> item('collect') >> item('algo') >> item('nstep')
        ),
        unroll_len_check=item('policy') >> item('collect') >> mcmp(item('unroll_len'), "<=", item('traj_len')),
        eps_check=item('policy') >> item('command') >> item('eps') >> mcmp(item('start'), ">=", item('end')),
        traj_len_check=mcmp(
            item('policy') >> item('collect') >> item('traj_len'), '==',
            item('actor') >> item('traj_len')
        )
    ) & keep()
    cartpole_dqn_loader = element_loader >> relation_loader

    try:
        output = cartpole_dqn_loader(cartpole_dqn_default_config)
        pretty_print(output, direct_print=True)
    except Exception as e:
        raise e
        assert False, "loader error"
