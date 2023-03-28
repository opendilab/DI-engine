from easydict import EasyDict

imagenet_res18_config = dict(
    exp_name='imagenet_res18',
    policy=dict(
        cuda=True,
        multi_gpu=True,
        learn=dict(
            bp_update_sync=True,
            train_epoch=200,
            batch_size=32,
            learning_rate=0.01,
            decay_epoch=30,
            decay_rate=0.1,
            warmup_lr=1e-4,
            warmup_epoch=3,
            weight_decay=1e-4,
            learner=dict(
                log_show_freq=10,
                hook=dict(
                    log_show_after_iter=int(1e9),  # use user-defined hook, disable it
                    save_ckpt_after_iter=1000,
                )
            )
        ),
        collect=dict(
            learn_data_path='/mnt/lustre/share/images/train',
            eval_data_path='/mnt/lustre/share/images/val',
        ),
        eval=dict(batch_size=32, evaluator=dict(eval_freq=1, stop_value=dict(loss=0.5, acc1=75.0, acc5=95.0))),
    ),
    env=dict(),
)
imagenet_res18_config = EasyDict(imagenet_res18_config)
