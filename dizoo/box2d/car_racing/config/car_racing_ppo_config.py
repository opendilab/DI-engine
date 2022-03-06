from easydict import EasyDict
from ding.entry import serial_pipeline_onpolicy

car_racing_ppo_config = dict(
    exp_name='car_racing_ppo',
    env=dict(
        env_id='CarRacing-v0',
        collector_env_num=8,
        evaluator_env_num=5,
        # (bool) Scale output action into legal range.
        act_scale=True,
        n_evaluator_episode=5,
        stop_value=300,
        rew_clip=True,
        replay_path=None,
        frame_stack=4,
        is_train=True,
        render=False,
    ),
    policy=dict(
        cuda=False,
        action_space='continuous',
        model=dict(
            action_space='continuous',
            obs_shape=[4, 84, 84],
            action_shape=4,
        ),
        learn=dict(
            epoch_per_collect=10,
            batch_size=64,
            learning_rate=0.001,
            value_weight=0.5,
            entropy_weight=0.01,
            clip_ratio=0.2,
            adv_norm=True,
        ),
        collect=dict(
            n_sample=2048,
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
    ),
)
car_racing_ppo_config = EasyDict(car_racing_ppo_config)
main_config = car_racing_ppo_config
car_racing_ppo_create_config = dict(
    env=dict(
        type='car_racing',
        import_names=['dizoo.box2d.car_racing.envs.car_racing_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='ppo'),
)
car_racing_ppo_create_config = EasyDict(car_racing_ppo_create_config)
create_config = car_racing_ppo_create_config

if __name__ == "__main__":
    serial_pipeline_onpolicy([main_config, create_config], seed=0)
