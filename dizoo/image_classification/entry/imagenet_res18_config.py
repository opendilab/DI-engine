from easydict import EasyDict

imagenet_res18_config = dict(
    exp_name='imagenet_res18',
    policy=dict(
        cuda=False,
        learn=dict(
            multi_gpu=False,
            train_epoch=200,
            batch_size=3,
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
            learn_data_path='./data/train',
            eval_data_path='./data/eval',
        ),
        eval=dict(batch_size=4, )
    ),
    env=dict(),
)
imagenet_res18_config = EasyDict(imagenet_res18_config)
