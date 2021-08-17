from easydict import EasyDict
from ding.entry import serial_pipeline, serial_pipeline_reward_model
mountaincar_dqn_config = dict(
    exp_name='mountaincar_dqn',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=195,
    ),
    policy=dict(
        cuda=False,
        random_collect_size=10000,
        model=dict(
            obs_shape=2,
            action_shape=3,
            encoder_hidden_size_list=[128, 128, 64],
            dueling=True,
        ),
        nstep=1,
        discount_factor=0.97,
        learn=dict(
            batch_size=64,
            learning_rate=0.001,
        ),
        collect=dict(n_sample=8),
        eval=dict(evaluator=dict(eval_freq=50, )),
        other=dict(
            eps=dict(
                type='exp',
                start=0.1,
                end=0.001,
                decay=10000,
            ),
            replay_buffer=dict(
                type='naive',
                replay_buffer_size=20000,
            ),
        ),
    ),
    reward_model=dict(
        type='countbased',
        counter_type='SimHash',
        bonus_coefficent=0.01,
        state_dim=2,
        hash_dim=64,
    ),
)
mountaincar_dqn_config = EasyDict(mountaincar_dqn_config)
main_config = mountaincar_dqn_config
mountaincar_dqn_create_config = dict(
    env=dict(
        type='mountaincar',
        import_names=['dizoo.classic_control.mountaincar.envs.mountaincar_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='dqn'),
)
mountaincar_dqn_create_config = EasyDict(mountaincar_dqn_create_config)
create_config = mountaincar_dqn_create_config

if __name__ == '__main__':
    serial_pipeline_reward_model([main_config, create_config], seed=0)
