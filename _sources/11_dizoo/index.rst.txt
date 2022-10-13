Learn From DI-zoo
===============================

What is DI-zoo
-------------------------------

DI-zoo is a collection of reinforcement learning environments wrapped with DI-engine. It covers the vast majority of reinforcement learning environments, including basic environments like `OpenAI Gym <https://gym.openai.com/>`_, as well as more complex environments such as `SMAC <https://github.com/oxwhirl/smac>`_. Besides, for each environment, DI-zoo provides the entries of different algorithms with their optimal configurations.


The structure of DI-zoo
-------------------------------

For a certain environment/policy pair, in order to run RL experiment in DI-engine, DI-zoo mainly provides two files: the ``config.py`` file, including the key configuration required as well as the entry point to run the RL experiment; the ``env.py`` file, containing the encapsulation of the environment to run in DI-engine.

.. note::
    
    Besides, some environment/policy pairs also possess a ``main.py`` entry file, which is the training pipeline left over from the previous version.

Here we briefly show the structure of DI-zoo based on the CartPole environment and DQN algorithm.

.. code-block::

  dizoo/
    classic_control/
      cartpole/
        config/cartpole_dqn_config.py # Config
        entry/cartpole_dqn_main.py  # Main 
        envs/cartpole_env.py  # Env

How to use DI-zoo
-------------------------------
You can directly execute the ``config.py`` file provided by DI-zoo to train a certain environment/policy pair. For CartPole/DQN, you can easily perform the RL experiment with the following code:

.. code-block:: bash

    python dizoo/classic_control/cartpole/config/cartpole_dqn_config.py

DI-engine also provides the CLI tool for users, you can type the following command in your terminal:

.. code-block:: bash

   ding -v

If the terminal returns the correct information, you can use this CLI tool for the common training and evaluation, and you can type ``ding -h`` for further usage。

To train CartPole/DQN, you can directly type the following command in the terminal:

.. code-block:: bash

   ding -m serial -c cartpole_dqn_config.py -s 0

where ``-m serial`` means that the training pipeline you call is ``serial_pipeline``. ``-c cartpole_dqn_config.py`` means that the ``config`` file you use is ``cartpole_dqn_config.py``. ``-s 0`` means ``seed`` is 0.

Customization of DI-zoo
-------------------------------

You can customize your training process or tune the performance of your RL experiment by changing the configuration in ``config.py``.

Here we use ``cartpole_dqn_config.py`` as an example: 

.. code-block:: python

    from easydict import EasyDict

    cartpole_dqn_config = dict(
        exp_name='cartpole_dqn_seed0',
        env=dict(
            collector_env_num=8,
            evaluator_env_num=5,
            n_evaluator_episode=5,
            stop_value=195,
            replay_path='cartpole_dqn_seed0/video',
        ),
        policy=dict(
            cuda=False,
            load_path='cartpole_dqn_seed0/ckpt/ckpt_best.pth.tar',  # necessary for eval
            model=dict(
                obs_shape=4,
                action_shape=2,
                encoder_hidden_size_list=[128, 128, 64],
                dueling=True,
            ),
            nstep=1,
            discount_factor=0.97,
            learn=dict(
                batch_size=64,
                learning_rate=0.001,
            ),
            collect=dict(n_sample=8),
            eval=dict(evaluator=dict(eval_freq=40, )),
            other=dict(
                eps=dict(
                    type='exp',
                    start=0.95,
                    end=0.1,
                    decay=10000,
                ),
                replay_buffer=dict(replay_buffer_size=20000, ),
            ),
        ),
    )
    cartpole_dqn_config = EasyDict(cartpole_dqn_config)
    main_config = cartpole_dqn_config
    cartpole_dqn_create_config = dict(
        env=dict(
            type='cartpole',
            import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
        ),
        env_manager=dict(type='base'),
        policy=dict(type='dqn'),
        replay_buffer=dict(
            type='deque',
            import_names=['ding.data.buffer.deque_buffer_wrapper']
        ),
    )
    cartpole_dqn_create_config = EasyDict(cartpole_dqn_create_config)
    create_config = cartpole_dqn_create_config

    if __name__ == "__main__":
        # or you can enter `ding -m serial -c cartpole_dqn_config.py -s 0`
        from ding.entry import serial_pipeline
        serial_pipeline((main_config, create_config), seed=0)

The two dictionary objects ``cartpole_dqn_config`` and ``cartpole_dqn_create_config`` contain the key configurations required for CartPole/DQN training. You can change the behavior of your training pipeline by changing the configuration here. For example, by changing ``cartpole_dqn_config.policy.cuda`` , you can choose whether to use your cuda device to run the entire training process.

If you want to use other training pipelines provided by DI-engine, or use your own customized training pipelines, you only need to change the ``__main__`` function of ``config.py`` that calls the training pipeline. For example, you can change the ``serial_pipeline`` in the example to ``parallel_pipeline`` to call the parallel training pipeline.

For the CLI tool ``ding``, you can also change the previous cli command to

.. code-block:: bash

   ding -m parallel -c cartpole_dqn_config.py -s 0

to call ``parallel_pipeline``.

.. note ::

    To customize the training pipeline, you can refer to `serial_pipeline <https://github.com/opendilab/DI-engine/blob/0fccfcb046f04767504f68220d96a6608bb38f29/ding/entry/serial_entry.py#L17>`_ , or refer to `DQN example <https://github.com/opendilab/DI-engine/blob/main/ding/example/dqn.py>`_, using the the `middleware <../03_system/middleware.html>`_ provided by DI-engine to build the pipeline.

    If you want to use your own environment in DI-engine, you can just inherit ``BaseEnv`` implemented by DI-engine. For this part you can refer to `How to migrate your environment to DI-engine <../04_best_practice/ding_env.html>`_

List of algorithms and environments supported by DI-zoo
---------------------------------------------------------

`The algorithm documentation of DI-engine <../12_policies/index.html>`_

`The environment documentation of DI-engine <../13_envs/index.html>`_

`List of supported algorithms <https://github.com/opendilab/DI-engine#algorithm-versatility>`_

`List of supported environments <https://github.com/opendilab/DI-engine#environment-versatility>`_
