from easydict import EasyDict
from ding.entry import serial_pipeline_max_entropy

lunarlander_ppo_config = dict(
    exp_name='lunarlander_guided_cost_F',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=200,
    ),
    reward_model=dict(
        learning_rate=0.001,
        input_size=9,
        batch_size=32,
        update_per_collect=20,
    ),
    policy=dict(
        cuda=False,
        on_policy = False,
        recompute_adv = True,
        model=dict(
            obs_shape=8,
            action_shape=4,
        ),
        learn=dict(
            update_per_collect=8,
            batch_size=800,
            learning_rate=0.001,
            value_weight=0.5,
            entropy_weight=0.01,
            clip_ratio=0.2,
            nstep=1,
            nstep_return=False,
            adv_norm=True,
        ),
        collect=dict(
            demonstration_info_path ='/home/SENSETIME/weiyuhong/Desktop/Guided_cost_0825/DI-engine/lunarlander_expert/ckpt/ckpt_best.pth.tar',
            n_sample=800,
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
    ),
)
lunarlander_ppo_config = EasyDict(lunarlander_ppo_config)
main_config = lunarlander_ppo_config
lunarlander_ppo_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='ppo_offpolicy'),
    reward_model=dict(type='max_entropy'),
)
lunarlander_ppo_create_config = EasyDict(lunarlander_ppo_create_config)
create_config = lunarlander_ppo_create_config

if __name__ == "__main__":
    serial_pipeline_max_entropy([main_config, create_config], seed=0)
