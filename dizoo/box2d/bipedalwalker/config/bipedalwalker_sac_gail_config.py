from easydict import EasyDict

obs_shape = 24
act_shape = 4
bipedalwalker_sac_gail_default_config = dict(
    exp_name='bipedalwalker_sac_gail_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        # (bool) Scale output action into legal range.
        act_scale=True,
        n_evaluator_episode=5,
        stop_value=300,
        rew_clip=True,
        # The path to save the game replay
        replay_path=None,
    ),
    reward_model=dict(
        type='gail',
        input_size=obs_shape + act_shape,
        hidden_size=64,
        batch_size=64,
        learning_rate=1e-3,
        update_per_collect=100,
        # Users should add their own expert data path here.
        # Expert data path should lead to a file to store data or load the stored data.
        # It is the path where the expert data is stored
        # Absolute path is recommended.
        # In DI-engine, it is usually located in ``exp_name`` directory
        expert_data_path='expert_data_path_placeholder',
        # Users should add their expert load path here. Expert load path should lead to a model.
        # It is the path where the expert state_dict is stored
        # Absolute path is recommended.
        # In DI-engine, it is ``exp_name/ckpt/ckpt_best.pth.tar``.
        expert_load_path='expert_load_path_placeholder',
        collect_count=100000,
        # Users should add their own load path here. Load path should lead to a model.
        # It is the path where the state_dict of reward model is stored
        # Absolute path is recommended.
        # In DI-engine, it is ``exp_name/reward_model/ckpt/ckpt_best.pth.tar``
        load_path='load_path_placeholder',
    ),
    policy=dict(
        cuda=False,
        priority=False,
        random_collect_size=1000,
        model=dict(
            obs_shape=obs_shape,
            action_shape=act_shape,
            twin_critic=True,
            action_space='reparameterization',
            actor_head_hidden_size=128,
            critic_head_hidden_size=128,
        ),
        learn=dict(
            update_per_collect=1,
            batch_size=128,
            learning_rate_q=0.001,
            learning_rate_policy=0.001,
            learning_rate_alpha=0.0003,
            ignore_done=True,
            target_theta=0.005,
            discount_factor=0.99,
            auto_alpha=True,
            value_network=False,
        ),
        collect=dict(
            n_sample=128,
            unroll_len=1,
        ),
        other=dict(replay_buffer=dict(replay_buffer_size=100000, ), ),
    ),
)
bipedalwalker_sac_gail_default_config = EasyDict(bipedalwalker_sac_gail_default_config)
main_config = bipedalwalker_sac_gail_default_config

bipedalwalker_sac_gail_create_config = dict(
    env=dict(
        type='bipedalwalker',
        import_names=['dizoo.box2d.bipedalwalker.envs.bipedalwalker_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='sac',
        import_names=['ding.policy.sac'],
    ),
    replay_buffer=dict(type='naive', ),
)
bipedalwalker_sac_gail_create_config = EasyDict(bipedalwalker_sac_gail_create_config)
create_config = bipedalwalker_sac_gail_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial_gail -c bipedalwalker_sac_gail_config.py -s 0`
    # then input the config you used to generate your expert model in the path mentioned above
    # e.g. bipedalwalker_sac_config.py
    from ding.entry import serial_pipeline_gail
    from dizoo.box2d.bipedalwalker.config import bipedalwalker_sac_config, bipedalwalker_sac_create_config
    expert_main_config = bipedalwalker_sac_config
    expert_create_config = bipedalwalker_sac_create_config
    serial_pipeline_gail(
        [main_config, create_config], [expert_main_config, expert_create_config], seed=0, collect_data=True
    )
