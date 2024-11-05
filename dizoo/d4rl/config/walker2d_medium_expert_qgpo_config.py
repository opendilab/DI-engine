from easydict import EasyDict

main_config = dict(
    exp_name='walker2d_medium_expert_v2_QGPO_seed0',
    env=dict(
        env_id="walker2d-medium-expert-v2",
        evaluator_env_num=8,
        n_evaluator_episode=8,
    ),
    dataset=dict(env_id="walker2d-medium-expert-v2", ),
    policy=dict(
        cuda=True,
        on_policy=False,
        #load_path='./walker2d_medium_expert_v2_QGPO_seed0/ckpt/iteration_600000.pth.tar',
        model=dict(
            qgpo_critic=dict(
                alpha=3,
                q_alpha=1,
            ),
            device='cuda',
            obs_dim=17,
            action_dim=6,
        ),
        learn=dict(
            learning_rate=1e-4,
            batch_size=4096,
            batch_size_q=256,
            M=16,
            diffusion_steps=15,
            behavior_policy_stop_training_iter=600000,
            energy_guided_policy_begin_training_iter=600000,
            q_value_stop_training_iter=1100000,
        ),
        eval=dict(
            guidance_scale=[0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0],
            diffusion_steps=15,
            evaluator=dict(eval_freq=50000, ),
        ),
    ),
)
main_config = EasyDict(main_config)

create_config = dict(
    env_manager=dict(type='base'),
    policy=dict(type='qgpo', ),
)
create_config = EasyDict(create_config)
