import math

import pytest

from app_zoo.classic_control.cartpole.config.cartpole_dqn_default_config import cartpole_dqn_default_config
from nervex.loader import dict_, is_type, to_type, collection, interval, is_positive, mcmp, enum, item, raw, check_only
from nervex.utils import pretty_print


@pytest.mark.unittest
def test_real_loader():
    element_loader = dict_(
        env=item('env') >> dict_(
            manager=item('manager') >> dict_(
                type=item('type') >> enum('base', 'subprocess', 'async_subprocess'),
                # shared_memory=item('shared_memory') >> is_type(bool),
                # context=item('context') >> enum('fork', 'spawn', 'forkserver') | raw('fork')
            ),
            env_kwargs=item('env_kwargs') >> dict_(
                import_names=item('import_names') >> collection(str),
                env_type=item('env_type') >> is_type(str),
                collector_env_num=item('collector_env_num') >> is_type(int) >> interval(1, 32),
                evaluator_env_num=item('evaluator_env_num') >> is_type(int) >> interval(1, 32),
            ),
        ),
        policy=item('policy') >> dict_(
            use_cuda=item('use_cuda') >> is_type(bool),
            policy_type=item('policy_type') >> is_type(str),
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
                unroll_len=item('unroll_len') >> is_type(int) >> interval(1, 200),
                algo=item('algo') >> dict_(nstep=item('nstep') >> (is_type(int) & interval(1, 10))),
            ),
            other=item('other') >> dict_(
                eps=item('eps') >> dict_(
                    type=item('type') >> enum('linear', 'exp'),
                    start=item('start') >> interval(0.0, 1.0, left_ok=False),
                    end=item('end') >> interval(0.0, 1.0, right_ok=False),
                    decay=item('decay') >> (is_type(int) | (is_type(float) >> to_type(int))) >> is_positive(),
                )
            ),
        ),
        replay_buffer=item('replay_buffer') >>
        dict_(replay_buffer_size=item('replay_buffer_size') >> is_type(int) >> interval(1, math.inf), ),
        learner=item('learner') >> dict_(load_path=item('load_path') >> is_type(str)),
        collector=item('collector') >> dict_(
            n_sample=item('n_sample') >> is_type(int) >> interval(8, 128),
            traj_len=item('traj_len') >> ((is_type(int) >> interval(1, 200)) | (enum("inf") >> to_type(float))),
            collect_print_freq=item('collect_print_freq') >> is_type(int) >> interval(1, 1000),
        ),
        evaluator=item('evaluator') >> dict_(
            n_episode=item('n_episode') >> is_type(int) >> interval(2, 10),
            eval_freq=item('eval_freq') >> is_type(int) >> interval(1, 500),
        ),
    )
    learn_nstep = item('policy') >> item('learn') >> item('algo') >> item('nstep')
    collect_nstep = item('policy') >> item('collect') >> item('algo') >> item('nstep')
    policy_unroll_len = item('policy') >> item('collect') >> item('unroll_len')
    collector_traj_len = item('collector') >> item('traj_len')
    relation_loader = check_only(
        dict_(
            nstep_check=mcmp(learn_nstep, "==", collect_nstep),
            unroll_len_check=mcmp(policy_unroll_len, "<=", collector_traj_len),
            eps_check=item('policy') >> item('other') >> item('eps') >> mcmp(item('start'), ">=", item('end')),
        )
    )
    cartpole_dqn_loader = element_loader >> relation_loader

    assert 'context' not in cartpole_dqn_default_config['env']['manager']
    output = cartpole_dqn_loader(cartpole_dqn_default_config)
    pretty_print(output, direct_print=True)
    # assert output['env']['manager']['context'] == 'fork'
