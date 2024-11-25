加载预训练模型与断点续训
========================

在使用 DI-engine 进行强化学习实验时，加载预训练的 ``ckpt`` 文件以实现断点续训是非常常见的需求。本文将以 ``cartpole_ppo_config.py`` 为例，详细说明如何使用 DI-engine 加载预训练模型并进行无缝的断点续训。

加载预训练模型
*****************

配置 ``load_ckpt_before_run``
------------------

要加载预训练模型，首先需要在配置文件中指定预训练的 ``ckpt`` 文件路径。该路径通过 ``load_ckpt_before_run`` 字段进行配置。

示例代码::

    from easydict import EasyDict
    
    cartpole_ppo_config = dict(
        exp_name='cartpole_ppo_seed0',
        env=dict(
            collector_env_num=8,
            evaluator_env_num=5,
            n_evaluator_episode=5,
            stop_value=195,
        ),
        policy=dict(
            cuda=False,
            action_space='discrete',
            model=dict(
                obs_shape=4,
                action_shape=2,
                action_space='discrete',
                encoder_hidden_size_list=[64, 64, 128],
                critic_head_hidden_size=128,
                actor_head_hidden_size=128,
            ),
            learn=dict(
                epoch_per_collect=2,
                batch_size=64,
                learning_rate=0.001,
                value_weight=0.5,
                entropy_weight=0.01,
                clip_ratio=0.2,
                # ======== Path to the pretrained checkpoint (ckpt) ========
                learner=dict(hook=dict(load_ckpt_before_run='/path/to/your/ckpt/iteration_100.pth.tar')),
            ),
            collect=dict(
                n_sample=256,
                unroll_len=1,
                discount_factor=0.9,
                gae_lambda=0.95,
            ),
            eval=dict(evaluator=dict(eval_freq=100, ), ),
        ),
    )
    cartpole_ppo_config = EasyDict(cartpole_ppo_config)
    main_config = cartpole_ppo_config

在上面的例子中，``load_ckpt_before_run`` 明确指定了预训练模型的路径 ``/path/to/your/ckpt/iteration_100.pth.tar``。当你运行这段代码时，DI-engine 会自动加载该路径下的模型权重，并在此基础上继续训练。

模型加载流程
------------

模型的加载流程主要发生在 `entry <https://github.com/opendilab/DI-engine/blob/main/ding/entry/>`_  路径下的主文件中，下面以 `serial_entry_onpolicy.py <https://github.com/opendilab/DI-engine/blob/main/ding/entry/serial_entry_onpolicy.py>`_ 文件为例进行说明。
加载预训练模型的关键操作是通过 DI-engine 的 ``hook`` 机制实现的：

.. code-block:: python

    # Learner's before_run hook.
    learner.call_hook('before_run')
    if cfg.policy.learn.get('resume_training', False):
        collector.envstep = learner.collector_envstep

当 ``load_ckpt_before_run`` 不为空时，DI-engine 会自动调用 ``learner`` 的 ``before_run`` 钩子函数来加载指定路径的预训练模型。具体实现代码可以参考 DI-engine 的 `learner_hook.py <https://github.com/opendilab/DI-engine/blob/main/ding/worker/learner/learner_hook.py#L86>`_。

其中，policy 本身的 checkpoint 保存和加载功能是通过 ``_load_state_dict_learn`` 和 ``_state_dict_learn`` 方法实现的。例如，PPO policy 中的实现位于以下位置：

- `PPO policy _load_state_dict_learn <https://github.com/opendilab/DI-engine/blob/main/ding/policy/ppo.py#L1827>`_
- `PPO policy _state_dict_learn <https://github.com/opendilab/DI-engine/blob/main/ding/policy/ppo.py#L1841>`_


断点续训
**********

续训日志与 TensorBoard 路径管理
------------------------------

在默认情况下，DI-engine 会为每次实验创建一个新的日志路径，以避免覆盖之前的训练数据和 TensorBoard 日志。如果你希望在断点续训时将日志与之前的实验保存在同一目录下，可以通过在配置文件中设置 ``resume_training=True`` (其默认值为 False) 来实现。

示例代码::

    learn=dict(
        ...  # 其他部分代码
        learner=dict(hook=dict(load_ckpt_before_run='/path/to/your/ckpt/iteration_100.pth.tar')),
        resume_training=True,
    )

当 ``resume_training=True`` 时，DI-engine 会将新的日志和 TensorBoard 数据保存在原来的路径下。

关键代码为::

    # 注意renew_dir 的默认值为True，当 resume_training=True 时，renew_dir 被设置为了 False，以保证日志路径的一致性
    cfg = compile_config(cfg, seed=seed, env=env_fn, auto=True, create_cfg=create_cfg, save_cfg=True, renew_dir=not cfg.policy.learn.get('resume_training', False))

同时，加载的 ``ckpt`` 文件中的 ``train_iter`` 和 ``collector.envstep`` 将被恢复，训练过程会从之前的训练断点无缝衔接。

续训的迭代/步数恢复
------------------

在断点续训时，训练的 ``iter`` 和 ``steps`` 将从加载的 ``ckpt`` 中保存的最后一次迭代和步数继续。通过这种方式，DI-engine 实现了训练过程的无缝衔接，确保了训练进度的准确性。

第一次训练 (pretrain) 结果：

下图显示了第一次训练 (pretrain) 的 ``evaluator`` 结果，分别以 ``iter`` 和 ``steps`` 为横轴：

        .. image:: images/cartpole_ppo_evaluator_iter_pretrain.png
            :align: center
            :scale: 40%

        .. image:: images/cartpole_ppo_evaluator_step_pretrain.png
            :align: center
            :scale: 40%

第二次训练 (resume) 结果：

下图显示了第二次训练 (resume) 的 ``evaluator`` 结果，分别以 ``iter`` 和 ``steps`` 为横轴：

        .. image:: images/cartpole_ppo_evaluator_iter_resume.png
            :align: center
            :scale: 40%

        .. image:: images/cartpole_ppo_evaluator_step_resume.png
            :align: center
            :scale: 40%

通过这些图表，能够明显看出训练在断点续训后从上次的状态继续进行，且评估指标在相同的迭代/步长下表现出一致性。

总结
*****

在使用 DI-engine 进行强化学习实验时，加载预训练模型和断点续训是实现长时间训练稳定性的重要手段。通过本文的示例与说明，我们可以看到：

1. **预训练模型加载** 是通过 ``load_ckpt_before_run`` 字段配置，并在训练前通过 ``hook`` 机制自动加载。
2. **断点续训** 可以通过设置 ``resume_training=True`` 来实现，确保日志和训练进度的无缝衔接。
3. 在实际实验中，合理管理日志路径和断点数据，可以避免重复训练和数据丢失，提高实验的效率与可重复性。

希望本文为你在 DI-engine 上的实验提供了清晰的操作指南。
