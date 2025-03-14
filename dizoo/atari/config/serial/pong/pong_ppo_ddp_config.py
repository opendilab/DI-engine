from easydict import EasyDict

pong_ppo_config = dict(
    exp_name='data_pong/pong_ppo_ddp_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=20,
        env_id='PongNoFrameskip-v4',
        #'ALE/Pong-v5' is available. But special setting is needed after gym make.
        frame_stack=4,
    ),
    policy=dict(
        multi_gpu=True,
        cuda=True,
        recompute_adv=True,
        action_space='discrete',
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=6,
            action_space='discrete',
            encoder_hidden_size_list=[64, 64, 128],
            actor_head_hidden_size=128,
            critic_head_hidden_size=128,
        ),
        learn=dict(
            epoch_per_collect=10,
            update_per_collect=1,
            batch_size=320,
            learning_rate=3e-4,
            value_weight=0.5,
            entropy_weight=0.001,
            clip_ratio=0.2,
            adv_norm=True,
            value_norm=True,
            # for ppo, when we recompute adv, we need the key done in data to split traj, so we must
            # use ignore_done=False here,
            # but when we add key traj_flag in data as the backup for key done, we could choose to use ignore_done=True
            # for halfcheetah, the length=1000
            ignore_done=False,
            grad_clip_type='clip_norm',
            grad_clip_value=0.5,
        ),
        collect=dict(
            n_sample=3200,
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
        eval=dict(evaluator=dict(eval_freq=1000, )),
    ),
)
main_config = EasyDict(pong_ppo_config)

pong_ppo_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo'),
)
create_config = EasyDict(pong_ppo_create_config)

if __name__ == "__main__":
    """
    Overview:
        This script should be executed with <nproc_per_node> GPUs.
        Run the following command to launch the script:
        python -m torch.distributed.launch --nproc_per_node=2 ./dizoo/atari/config/serial/pong/pong_ppo_ddp_config.py
    """
    from ding.utils import DDPContext
    from ding.entry import serial_pipeline_onpolicy
    with DDPContext():
        serial_pipeline_onpolicy((main_config, create_config), seed=0, max_env_step=int(3e6))
