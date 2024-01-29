from easydict import EasyDict

tabmwp_prompt_pg_config = dict(
    exp_name='tabmwp_prompt_pg_seed0',
    env=dict(
        collector_env_num=1,
        evaluator_env_num=1,
        n_evaluator_episode=1,
        stop_value=1,
        cand_number=16,
        train_number=80,
        engine='text-davinci-002',
        temperature=0.,
        max_tokens=512,
        top_p=1.,
        frequency_penalty=0.,
        presence_penalty=0.,
        option_inds=["A", "B", "C", "D", "E", "F"],
        # The API-key of openai. You can get your key in this website: https://platform.openai.com/
        api_key='',
        enable_replay=True,
        prompt_format='TQ-A',
        seed=0,
    ),
    policy=dict(
        cuda=True,
        shot_number=2,
        model=dict(
            model_name="bert-base-uncased",
            add_linear=True,
            freeze_encoder=True,
            embedding_size=128,
        ),
        learn=dict(
            batch_size=10,
            # (bool) Whether to normalize advantage. Default to False.
            learning_rate=0.001,
            # (float) loss weight of the value network, the weight of policy network is set to 1
            entropy_weight=0.001,
            weight_decay=5e-3,
            grad_norm=0.5,
        ),
        collect=dict(
            # (int) collect n_sample data, train model 1 times
            n_sample=20,
            discount_factor=0.,
        ),
        eval=dict(evaluator=dict(eval_freq=500, )),
    ),
)
main_config = EasyDict(tabmwp_prompt_pg_config)

tabmwp_prompt_pg_config = dict(
    env=dict(
        type='tabmwp',
        import_names=['dizoo.tabmwp.envs.tabmwp_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='prompt_pg'),
    replay_buffer=dict(type='naive'),
)
create_config = EasyDict(tabmwp_prompt_pg_config)

if __name__ == '__main__':
    from ding.entry import serial_pipeline_onpolicy
    serial_pipeline_onpolicy((main_config, create_config), seed=0)
