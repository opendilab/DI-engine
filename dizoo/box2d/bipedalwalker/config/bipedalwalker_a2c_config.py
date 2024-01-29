from easydict import EasyDict

bipedalwalker_a2c_config = dict(
    exp_name='bipedalwalker_a2c_seed0',
    env=dict(
        env_id='BipedalWalker-v3',
        collector_env_num=8,
        evaluator_env_num=8,
        # (bool) Scale output action into legal range.
        act_scale=True,
        n_evaluator_episode=8,
        stop_value=300,
        rew_clip=True,
        # The path to save the game replay
        # replay_path='./bipedalwalker_a2c_seed0/video',
    ),
    policy=dict(
        cuda=True,
        # load_path="./bipedalwalker_a2c_seed0/ckpt/ckpt_best.pth.tar",
        action_space='continuous',
        model=dict(
            action_space='continuous',
            obs_shape=24,
            action_shape=4,
        ),
        learn=dict(
            # (int) the number of data for a train iteration
            batch_size=256,
            learning_rate=0.0003,
            # (float) loss weight of the value network, the weight of policy network is set to 1
            value_weight=0.5,
            # (float) loss weight of the entropy regularization, the weight of policy network is set to 1
            entropy_weight=0.001,
            # (float) discount factor for future reward, defaults int [0, 1]
            discount_factor=0.99,
            adv_norm=True,
        ),
        collect=dict(
            # (int) collect n_sample data, train model n_iteration times
            n_sample=512,
            discount_factor=0.99,
            collector=dict(collect_print_freq=100, ),
        ),
        eval=dict(evaluator=dict(eval_freq=100, )),
    ),
)
bipedalwalker_a2c_config = EasyDict(bipedalwalker_a2c_config)
main_config = bipedalwalker_a2c_config
bipedalwalker_a2c_create_config = dict(
    env=dict(
        type='bipedalwalker',
        import_names=['dizoo.box2d.bipedalwalker.envs.bipedalwalker_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='a2c'),
    replay_buffer=dict(type='naive'),
)
bipedalwalker_a2c_create_config = EasyDict(bipedalwalker_a2c_create_config)
create_config = bipedalwalker_a2c_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial_onpolicy -c bipedalwalker_a2c_config.py -s 0`
    from ding.entry import serial_pipeline_onpolicy
    serial_pipeline_onpolicy([main_config, create_config], seed=0)
