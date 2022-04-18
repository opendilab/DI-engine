from easydict import EasyDict

lunarlander_sqil_config = dict(
    exp_name='lunarlander_sqil_seed0',
    env=dict(
        # Whether to use shared memory. Only effective if "env_manager_type" is 'subprocess'
        collector_env_num=8,
        evaluator_env_num=8,
        env_id='LunarLander-v2',
        n_evaluator_episode=8,
        stop_value=200,
    ),
    policy=dict(
        cuda=False,
        model=dict(
            obs_shape=8,
            action_shape=4,
            encoder_hidden_size_list=[128, 128, 64],
            dueling=True,
        ),
        nstep=1,
        discount_factor=0.97,
        learn=dict(batch_size=64, learning_rate=0.001, alpha=0.08),
        collect=dict(
            n_sample=64,
            # Users should add their own model path here. Model path should lead to a model.
            # Absolute path is recommended.
            # In DI-engine, it is ``exp_name/ckpt/ckpt_best.pth.tar``.
            model_path='model_path_placeholder',
            # Cut trajectories into pieces with length "unrol_len".
            unroll_len=1,
        ),
        eval=dict(evaluator=dict(eval_freq=50, )),  # note: this is the times after which you learns to evaluate
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=10000,
            ),
            replay_buffer=dict(replay_buffer_size=20000, ),
        ),
    ),
)
lunarlander_sqil_config = EasyDict(lunarlander_sqil_config)
main_config = lunarlander_sqil_config
lunarlander_sqil_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='sql'),
)
lunarlander_sqil_create_config = EasyDict(lunarlander_sqil_create_config)
create_config = lunarlander_sqil_create_config

if __name__ == '__main__':
    # or you can enter `ding -m serial_sqil -c lunarlander_sqil_config.py -s 0`
    # then input the config you used to generate your expert model in the path mentioned above
    # e.g. spaceinvaders_dqn_config.py
    from ding.entry import serial_pipeline_sqil
    from dizoo.box2d.lunarlander.config import lunarlander_dqn_config, lunarlander_dqn_create_config
    expert_main_config = lunarlander_dqn_config
    expert_create_config = lunarlander_dqn_create_config
    serial_pipeline_sqil([main_config, create_config], [expert_main_config, expert_create_config], seed=0)