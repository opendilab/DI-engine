Specifications of DI-engine Config
==================================

In order to ensure the ease of use, readability, and scalability of config, the config submitted by developers needs to comply with the following specifications.

The config of DI-engine consists of two parts: main_config and create_config. 

Example Link
--------------

Example of Deep Q-Network (DQN)：

https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/config/serial/pong/pong_dqn_config.py

Example config for algorithms with model or data, e.g. SQIL:
https://github.com/opendilab/DI-engine/blob/main/dizoo/atari/config/serial/pong/pong_sqil_config.py

Details of the specification
------------------------------

Specification of Grammar
~~~~~~~~~~~~~~~~~~~~~~~~~

-  config needs to pass python's flake8 syntax check and execute yapf formatting.

Specification of naming 
~~~~~~~~~~~~~~~~~~~~~~~~

-  The file name of X_config.py, related variable name in main_config and create_config

   -  X_config.py is uniformly named in the format of <environment_name>_<algorithm_name>_config.py.
      The name of the X_config.py and related variable names in the config file do not need to add the default field. For example, the config file name hopper_onppo_default_config.py should be changed into hopper_onppo_config.py.

   -  Similarly
      For algorithms like ICM, the full algorithm is the module proposed in the paper combined with a baseline algorithm, its corresponding config name should be named as <env_name>_<module_name>_<baseline_name>_config.py, e.g. cartpole_icm_offppo_config.py
      ,such as cartpole_icm_offppo_config.py

   -  If the algorithm has verious versions including on-policy and off-policy, please unify the related name in X_config.py file name and related varible names in the file, and use onppo/offppo to distinguish on-policy and off-policy versions of the algorithm. For example, for the config of the on policy PPO algorithm, hopper_ppo_config.py should be changed to hopper_onppo_config.py.

-  exp_name field

   -  main_config must include exp_name filed

   -  The naming format is <environemnt>_<algorithm>_seed0, e.g. qbert_sqil_seed0
-  Name of the file path

   -  Please refer to the sqil example, commented accordingly. If multiple models need to be loaded, the model path variable is named as follows：prefix1_model_path, prefix2_model_path, .... The varibles of data_path are named in same way.

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

Main Specification
~~~~~~~~~~~~~~~~~~~~

-  For env_manager field in create_config, except for some simple environments such as cartpole, pendulum, bitflip
we set it as base, for other environments, we seet env_manager as subprocess：

   .. code:: python

      env_manager=dict(type='subprocess'),

-  Ensure evaluator_env_num：n_evaluator_episode = 1:1 （expect smac environment）

-  manager field shoudl generally not be included in the env field of main_config
(shared_memory defaults to True when the manager field is not included)：

   -  smac environment is an exception,due to the state dimension problem,smac needs to set shared_memory=Fasle.

   -  In environments other than the SMAC environment, if an error is reported due to the state dimension problem, you can include manager field and set shared_memory=False.

-  If you want to turn on/off shared memory, please control it in env.manager filed

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

   -  iin env field, we have two fields: type and import_names :
      Such as：

   .. code:: python

      env=dict(
          type='atari',
          import_names=['dizoo.atari.envs.atari_env'],
      ),

   -  Generally speaking, the field replay_buffer is unnecessary. But if you want to use the buffer stored as deque，you can specify the type of replay_buffer in following way：

      .. code::

         replay_buffer=dict(type='deque'),

-  serial_pipeline

   -  Please apple secondary references to avoid circular
      import：use \ ``from ding.entry import serial_pipeline``\ instead of \ ``from ding.entry.serial_entry import serial_pipeline``

   -  Use\ ``[main_config, create_config]``
      to unify the style,If an algorithm needs to call other config,this convention can be waived。Such as imitation
      learning algorithm needs to introduce expert config, see the example of sqil for details。

   -  Each config must have a startup command written in a format similar to the following:

      .. code:: python

         if ___name___ == "___main___":
             # or you can enter `ding -m serial -c cartpole_dqn_config.py -s 0`
             from ding.entry import serial_pipeline
             serial_pipeline([main_config, create_config], seed=0)

      -  Remember this line from ding.entry import serial_pipeline should not as the head of the file,
but put it below if ___name___ == "___main___"::

   -  If the algorithm use different serial_pipeline_X,
      you need to add corresponding starting command ``serial_X``\ in \ https://github.com/opendilab/DI-engine/blob/5d2beed4a8a07fb70599d910c6d53cf5157b133b/ding/entry/cli.py#L189\ .

-  seed is set in the entry function, do not include seed in config.

-  If the hyperparameters in the algorithm have a certain reasonable range, please write a comment on the corresponding hyperparameters of the algorithm config. For instance the alpha value of sqil：

   .. code:: python

      alpha=0.1,  # alpha: 0.08-0.12

-  Please make sure all parameters in config are valid, unused redundant parameters should be deleted.

-  TODO is usually not included in the config, if you do need to write the TODO term, please clearly indicate the developer and content, e.g. TODO(name): xxx.
