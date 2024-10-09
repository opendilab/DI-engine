# You can conduct Experiments on D4RL with this config file through the following command:
# cd ../entry && python d4rl_qtransformer_main.py
from easydict import EasyDict
from ding.model import QTransformer


num_timesteps = 1

main_config = dict(
    exp_name="walker2d_qtransformer_online",
    env=dict(
        env_id="Walker2d-v3",
        norm_obs=dict(
            use_norm=False,
        ),
        norm_reward=dict(
            use_norm=False,
        ),
        collector_env_num=1,
        evaluator_env_num=4,
        stop_value=6000,
    ),
    # dataset=dict(
    #     dataset_folder="/root/code/DI-engine/qtransformer/model",
    #     num_timesteps=num_timesteps,
    # ),
    policy=dict(
        cuda=True,
        random_collect_size=10000,
        wandb=dict(project=f"Qtransformer_walker2d_{num_timesteps}"),
        model=dict(
            num_timesteps=num_timesteps,
            state_dim=17,
            action_dim=6,
            action_bin=256,
        ),
        learn=dict(
            update_per_collect=5,
            batch_size=256,
            learning_rate_q=3e-4,
            learning_rate_policy=1e-4,
            learning_rate_alpha=1e-4,
            ignore_done=False,
            target_theta=0.005,
            discount_factor=0.99,
            alpha=0.2,
            reparameterization=True,
            auto_alpha=False,
            # min_reward=0.0,
            # auto_alpha=False,
            # lagrange_thresh=-1.0,
            # min_q_weight=5.0,
        ),
        collect=dict(
            n_sample=1,
            unroll_len=1,
        ),
        command=dict(),
        eval=dict(
            evaluator=dict(
                eval_freq=10,
            )
        ),
        other=dict(
            replay_buffer=dict(
                replay_buffer_size=100000,
            ),
        ),
    ),
)

main_config = EasyDict(main_config)
main_config = main_config

create_config = dict(
    env=dict(
        type="mujoco",
        import_names=["dizoo.mujoco.envs.mujoco_env"],
    ),
    env_manager=dict(type="subprocess"),
    policy=dict(
        type="qtransformer",
        import_names=["ding.policy.qtransformer"],
    ),
    replay_buffer=dict(
        type="naive",
    ),
)
create_config = EasyDict(create_config)
create_config = create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c walker2d_sac_config.py -s 0`
    from qtransformer.algorithm.serial_entry import serial_pipeline

    model = QTransformer(**main_config.policy.model)
    serial_pipeline([main_config, create_config], seed=0, model=model)
