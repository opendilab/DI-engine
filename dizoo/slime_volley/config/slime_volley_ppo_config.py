from easydict import EasyDict
from ding.entry import serial_pipeline_onpolicy

slime_volley_ppo_config = dict(
    exp_name='slime_volley_ppo_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        agent_vs_agent=False,  # vs bot
        stop_value=5,  # 5 times per episode
        env_id="SlimeVolley-v0",
    ),
    policy=dict(
        cuda=True,
        action_space='discrete',
        model=dict(
            obs_shape=12,
            action_shape=6,
            action_space='discrete',
            encoder_hidden_size_list=[64, 64],
            critic_head_hidden_size=64,
            actor_head_hidden_size=64,
            share_encoder=False,  # It is not wise to share encoder in low-dimension observation.
        ),
        learn=dict(
            epoch_per_collect=5,
            batch_size=64,
            learning_rate=3e-4,
            entropy_weight=0.0,  # [0.01, 0.0]
        ),
        collect=dict(
            n_sample=4096,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
    ),
)
slime_volley_ppo_config = EasyDict(slime_volley_ppo_config)
main_config = slime_volley_ppo_config
slime_volley_ppo_create_config = dict(
    env=dict(
        type='slime_volley',
        import_names=['dizoo.slime_volley.envs.slime_volley_env'],
    ),
    env_manager=dict(type='subprocess'),  # if you want to save replay, it must use base
    policy=dict(type='ppo'),
)
slime_volley_ppo_create_config = EasyDict(slime_volley_ppo_create_config)
create_config = slime_volley_ppo_create_config

if __name__ == "__main__":
    serial_pipeline_onpolicy([main_config, create_config], seed=0)
