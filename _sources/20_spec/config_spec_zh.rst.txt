DI-engine Config规范
====================

开发者提交的 config 需要遵守以下规范，以保证 config 的易用性，可读性，与可扩展性。

DI-engine 的 config 包括 main_config 和 create_config 两部分。

示例链接
--------

普通算法 (DQN) 的示例：

https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/config/serial/pong/pong_dqn_config.py

包含模型或数据的算法 (SQIL) 的示例:

https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/config/serial/pong/pong_sqil_config.py

规范内容
--------

语法规范
~~~~~~~~

-  config 需要满足 flake8 python 语法检查, 以及进行 yapf 格式化。

命名规范
~~~~~~~~

-  config.py 文件名，main_config 和 create_config 相关变量名

   -  统一以<环境名>\_<算法名>\_config.py
      命名。文件的名称以及文件中相关变量名不用添加 default 字段。例如应该将文件名 hopper_onppo_default_config.py 改为 hopper_onppo_config.py。

   -  类似
      ICM 算法这种，总的算法是论文提出的模块再结合某个基线算法，其对应的 config 名称，按照<环境名>\_<模块名>\_<基线算法名>\_config.py
      命名，例如 cartpole_icm_offppo_config.py

   -  算法如果有 on-policy 和 off-policy 的不同版本，统一在 config.py 文件名和文件中相关变量名，使用 onppo/offppo 区分 on-policy 和 off-policy 版的算法。例如对于 PPO 算法的 config,
      应该将 hopper_ppo_config.py 改成 hopper_onppo_config.py。

-  exp_name 字段

   -  main_config 中必须添加 exp_name 字段

   -  命名规范为环境+算法+seed，例如\ ``qbert_sqil_seed0``

-  文件路径名

   -  参见 sqil 示例，并加上相应的注释。如果需要加载多个模型 model，则模型路径 (model_path) 变量按照以下方式命名：prefix1_model_path，prefix2_model_path，...,
      数据路径 (data_path) 变量命名类似。

.. code:: python

   config = dict(
       ...,
       # Users should add their own model path here. Model path should lead to a model.
       # Absolute path is recommended.
       # In DI-engine, it is ``exp_name/ckpt/ckpt_best.pth.tar``.
       model_path='model_path_placeholder',
       # Users should add their own data path here. Data path should lead to a file to store data or load the stored data.
       # Absolute path is recommended.
       # In DI-engine, it is usually located in ``exp_name`` directory
       data_path='data_path_placeholder',
   )

主要规范
~~~~~~~~

-  对于 create_config 中的 env_manager 字段，除了简单环境
   cartpole, pendulum, bitflip
   环境使用 base, 其他环境一般使用 subprocess：

   .. code:: python

      env_manager=dict(type='subprocess'),

-  保证 evaluator_env_num：n_evaluator_episode = 1:1 （ smac 环境例外）

-  在 main_config 的 env 字段中一般不应该包含 manager 字段
   (当不包含 manager 字段时，shared_memory 默认为 True)：

   -  smac 环境例外，由于状态维度问题，smac 需要设置 shared_memory=Fasle。

   -  smac 环境外的其他环境，如果由于状态维度问题运行报错，可以包含 manager 字段并设置 shared
      memory=False。

-  如果想开启/关闭 shared memory, 请在env.manager字段中进行控制

   .. code:: python

      config = dict(
          ...,
          env=dict(
              manager=dict(
                  shared_memory=True,
              ),
          ),
      )

-  create config

   -  env字段中，只需要包含 ``type`` 和 ``import_names``\ 两个字段,
      例如：

   .. code:: python

      env=dict(
          type='atari',
          import_names=['dizoo.atari.envs.atari_env'],
      ),

   -  一般不需要\ ``replay_buffer``\ 字段。如果想使用存储为deque的buffer，请在create_config中指定replay_buffer的type为deque：

      .. code::

         replay_buffer=dict(type='deque'),

-  serial_pipeline

   -  使用二级引用避免 circular
      import：即使用\ ``from ding.entry import serial_pipeline``\ 而不是\ ``from ding.entry.serial_entry import serial_pipeline``

   -  使用\ ``[main_config, create_config]``
      以统一风格，如果算法需要调用其他 config，可以不遵循此约定。例如 imitation
      learning 算法需要引入专家 config，具体参见 sqil 的示例。

   -  每一个 config 必须有一个启动命令，且写成类似下面这种格式

      .. code:: python

         if ___name___ == "___main___":
             # or you can enter `ding -m serial -c cartpole_dqn_config.py -s 0`
             from ding.entry import serial_pipeline
             serial_pipeline([main_config, create_config], seed=0)

      -  注意\ ``from ding.entry import serial_pipeline``\ 这行不要写在文件开头，
         要写在\ ``if ___name___ == "___main___":``\ 下面。

   -  如果算法使用了不同的 serial_pipeline_X，
      需要在\ https://github.com/opendilab/DI-engine/blob/5d2beed4a8a07fb70599d910c6d53cf5157b133b/ding/entry/cli.py#L189\ 中添加相应的启动命令对应
      ``serial_X``\ 。

-  seed 在入口函数中设置，config 中不要包含 seed。

-  如果算法中超参数有确定的一个合理范围，请在算法 config 的对应超参数上写上注释，例如 sqil 中的 alpha 值：

   .. code:: python

      alpha=0.1,  # alpha: 0.08-0.12

-  确保 config 中所有参数都是有效的，需要删除没有用到的 key。

-  一般在 config 中不包含 TODO 项，如果确实有必要写进 config，需要写清楚内容，例如：TODO(name):
   xxx.
