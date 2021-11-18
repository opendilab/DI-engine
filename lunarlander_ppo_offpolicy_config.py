from easydict import EasyDict
from ding.entry import serial_pipeline_max_entropy

cartpole_ppo_offpolicy_config = dict(
    exp_name='lunarlander_guided_cost',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=195,
    ),
    reward_model=dict(
        learning_rate=0.001,
        input_size=5,
        batch_size=32,
        update_per_collect=10,
    ),
    policy=dict(
        on_policy=False,
        cuda=False,
        recompute_adv = True,
        model=dict(
            obs_shape=4,
            action_shape=2,
            encoder_hidden_size_list=[64, 64, 128],
            critic_head_hidden_size=128,
            actor_head_hidden_size=128,
        ),
        learn=dict(
            update_per_collect=6,
            batch_size=64,
            learning_rate=0.001,
            value_weight=0.5,
            entropy_weight=0.01,
            clip_ratio=0.2,
        ),
        collect=dict(
            demonstration_info_path ='/home/SENSETIME/weiyuhong/Desktop/Guided_cost_0825(copy)/DI-engine/cartpole_ppo_offpolicy/ckpt/ckpt_best.pth.tar',
            n_sample=128,
            unroll_len=1,
            discount_factor=0.9,
            gae_lambda=0.95,
        ),
        other=dict(replay_buffer=dict(replay_buffer_size=5000))
    ),
)
cartpole_ppo_offpolicy_config = EasyDict(cartpole_ppo_offpolicy_config)
main_config = cartpole_ppo_offpolicy_config
cartpole_ppo_offpolicy_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='ppo_offpolicy'),
    reward_model=dict(type='max_entropy'),
)
cartpole_ppo_offpolicy_create_config = EasyDict(cartpole_ppo_offpolicy_create_config)
create_config = cartpole_ppo_offpolicy_create_config

if __name__ == "__main__":
    serial_pipeline_max_entropy([main_config, create_config], seed=0)