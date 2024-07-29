from easydict import EasyDict
import torch.nn as nn

ant_ppo_config = dict(
    exp_name="ant_onppo_seed0",
    env=dict(
        env_id='Ant-v3',
        collector_env_num=10,
        evaluator_env_num=10,
        n_evaluator_episode=10,
        stop_value=6000,
    ),
    policy=dict(
        cuda=True,
        recompute_adv=True,
        action_space='continuous',
        model=dict(
            encoder_hidden_size_list=[128, 128],
            action_space='continuous',
            obs_shape=111,
            action_shape=8,
            share_encoder=False,
            actor_head_layer_num=0,
            critic_head_layer_num=2,
            critic_head_hidden_size=256,
            actor_head_hidden_size=128,
            activation=nn.Tanh(),
            bound_type='tanh',
        ),
        learn=dict(
            epoch_per_collect=10,
            update_per_collect=1,
            batch_size=128,
            learning_rate=3e-4,
            lr_scheduler=dict(epoch_num=1500, min_lr_lambda=0),
            value_weight=0.5,
            entropy_weight=0.001,
            clip_ratio=0.2,
            adv_norm=True,
            value_norm=True,
            # When we recompute advantage, we need the key done in data to split trajectories, so we must
            # use 'ignore_done=False' here, but when we add key 'traj_flag' in data as the backup for key done,
            # we could choose to use 'ignore_done=True'. 'traj_flag' indicates termination of trajectory.
            ignore_done=False,
            grad_clip_type='clip_norm',
            grad_clip_value=0.5,
        ),
        collect=dict(
            n_sample=2048,
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
        eval=dict(evaluator=dict(eval_freq=5000, )),
    ),
)
ant_ppo_config = EasyDict(ant_ppo_config)
main_config = ant_ppo_config

ant_ppo_create_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo'),
)
ant_ppo_create_config = EasyDict(ant_ppo_create_config)
create_config = ant_ppo_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial_onpolicy -c ant_onppo_config.py -s 0 --env-step 1e7`
    from ding.entry import serial_pipeline_onpolicy
    serial_pipeline_onpolicy((main_config, create_config), seed=0)
