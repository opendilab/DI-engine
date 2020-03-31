Tutorial
===============================

.. toctree::
   :maxdepth: 2


Training
---------

1. Make sure that you have built SenseStar

2. Copy experiment dir(must be in dir `sc2learner`)

.. code-block:: bash

    cp -r experiment/ppo_baseline experiment/ppo_xxx

3. Modify config file into your experiment setting

  - config.yaml (training config, for learner and actor, especially specify your own learner_ip)
  - learner.sh (launch learner)
    - load_path(optional): checkpoint or pretrained model load path
    - data_load_path(optional): offline generated data load path
  - actor.sh (launch actor)

4. Start training your agent (single learner and multi actor)

.. code-block:: bash

    ./experiments/ppo_xxx/actor.sh <partition_name> <actor_num>

    ./experiments/ppo_xxx/learner.sh <partition_name>

5. log and viz

  - experiment/ppo_xxx/default_logger.txt (train logger)
  - experiment/ppo_xxx/checkpoints (checkpoints dir)
  - experiment/ppo_xxx/data (actor generated data dir, if 'save_data=True')
  - viz.sh(in dir 'sc2learner') can open tensorboard

.. code-block:: bash

    # usage
    ./viz.sh <port_id>
    # enter '<lustre_ip>:<port_id>' in your browser

You can use **sinfo** to inspect available partitions.

Evaluate by elite bot
---------------------
TBD


Test environments and interfaces
--------------------------------

1. Prepare config file

    modify `experiments/random_agent/eval.yaml`

.. code-block:: yaml

    common:
        config_name: eval
        num_episodes: 10  # the number of games per evaluate task launches
        agent: random  # [ppo, dqn, random, keyboard], default use random as test agent
        use_multiprocessing: False  # whether use multiprocessing in single task
    env:
        game_version: '4.10'
        map_name: AbyssalReef  # default map
    # the other part is omitted, if necessary, you can find them in eval.yaml


2. Start testing (random agent VS elite bot)

.. code-block:: bash

    ./experiments/random_agent/eval.sh <partition_name>

.. note::

    the min number of CPUs per game task need is 1, and the more CPUs it utilizes, the faster the game simulates.
