from easydict import EasyDict
from ding.entry import serial_pipeline
collector_env_num = 8
evaluator_env_num = 5
pong_r2d2_config = dict(
    exp_name='pong_r2d2_bs20_n5_ul80_upc8_tuf2500_ed1e4_rbs1e5_p',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=5,
        stop_value=20,
        env_id='PongNoFrameskip-v4',
        frame_stack=4,
    ),
    policy=dict(
        cuda=True,
        on_policy=False,
        priority=True,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=6,
            encoder_hidden_size_list=[128, 128, 512],
        ),
        discount_factor=0.997,
        burnin_step=20,
        nstep=5,
        # (int) the whole sequence length to unroll the RNN network minus
        # the timesteps of burnin part,
        # i.e., <the whole sequence length> = <burnin_step> + <unroll_len>
        unroll_len=80,
        learn=dict(
            # according to the R2D2 paper, actor parameter update interval is 400
            # environment timesteps, and in per collect phase, we collect 32 sequence
            # samples, the length of each samlpe sequence is <burnin_step> + <unroll_len>,
            # which is 100 in our seeting, 32*100/400=8, so we set update_per_collect=8
            # in most environments
            update_per_collect=8,
            batch_size=64,
            learning_rate=0.0005,
            target_update_freq=2500,
        ),
        collect=dict(
            each_iter_n_sample=32,
            env_num=collector_env_num,
        ),
        eval=dict(env_num=evaluator_env_num, ),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.05,
                decay=10000,
            ), replay_buffer=dict(replay_buffer_size=100000, )
        ),
    ),
)
pong_r2d2_config = EasyDict(pong_r2d2_config)
main_config = pong_r2d2_config
pong_r2d2_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='r2d2'),
)
pong_r2d2_create_config = EasyDict(pong_r2d2_create_config)
create_config = pong_r2d2_create_config

if __name__ == "__main__":
    serial_pipeline([main_config, create_config], seed=0)
