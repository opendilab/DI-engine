from easydict import EasyDict

base_learner_default_config = dict(
    load_path='',
    use_distributed=False,
    dataloader=dict(
        batch_size=2,
        chunk_size=2,
        num_workers=0,
    ),
    # --- Hooks ---
    hook=dict(
        load_ckpt=dict(
            name='load_ckpt',
            type='load_ckpt',
            priority=20,
            position='before_run',
        ),
        log_show=dict(
            name='log_show',
            type='log_show',
            priority=20,
            position='after_iter',
            ext_args=dict(freq=1),
        ),
        save_ckpt_after_run=dict(
            name='save_ckpt_after_run',
            type='save_ckpt',
            priority=20,
            position='after_run',
        )
    ),
)
base_learner_default_config = EasyDict(base_learner_default_config)
