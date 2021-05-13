import pytest
from easydict import EasyDict
from nervex.worker.coordinator import OneVsOneCommander



@pytest.fixture(scope='module')
def setup_1v1commander():
    nstep = 1
    eval_interval = 5
    __policy_default_config = dict(
        use_cuda=False,
        policy_type='dqn',
        model=dict(
            obs_dim=[4, 84, 84],
            action_dim=3,
            embedding_dim=64,
            encoder_kwargs=dict(encoder_type='conv2d'),
        ),
        learn=dict(
            batch_size=32,
            learning_rate=0.0001,
            weight_decay=0.,
            algo=dict(
                target_update_freq=500,
                discount_factor=0.99,
                nstep=nstep,
            ),
        ),
        collect=dict(
            traj_len=15,
            unroll_len=1,
            algo=dict(nstep=nstep),
        ),
        other=dict(eps=dict(
            type='linear',
            start=1.,
            end=0.005,
            decay=1000000,
        ), ),
    )
    config = dict(
        parallel_commander_type='one_vs_one',
        collector_task_space=2,
        learner_task_space=1,
        max_iterations=int(1e9),
        learner_cfg=dict(),
        collector_cfg=dict(
            env_kwargs=dict(eval_stop_value=20),
        ),
        replay_buffer_cfg=dict(
            replay_buffer_size=100000,
        ),
        policy=__policy_default_config,
        
        eval_interval=eval_interval,
        league=dict(
            league_type="one_vs_one",
        ),
    )
    config = EasyDict(config)
    return OneVsOneCommander(config)
