Tutorial
===============================

.. toctree::
   :maxdepth: 2


Training
---------

1. Make sure that you have built SenseStar

2. Copy experiment dir(must be in dir 'sc2learner')

.. code-block:: bash

    cp -r experiment/ppo_baseline experiment/ppo_xxx

3. Modify config file into your experiment setting
  - config.yaml (training config, for learner and actor, especially specify your own learner_ip)
  - learner.sh (learner train)

    - load_path: checkpoint or pretrained model load path
    - data_load_path: offline generated data load path

  - actor.sh (actor train)

4. Start training your agent (single learner and multi actor)

.. code-block:: bash

    ./experiments/ppo_baseline/actor.sh <partition_name> <actor_num>

    ./experiments/ppo_baseline/learner.sh <partition_name>

5. log and viz

  - viz.sh(in dir 'sc2learner') can open tensorboard
    - usage:
      - run './viz.sh <port_id>'
      - enter '<lustre_ip>:<port_id>' in your browser
  - experiment/ppo_xxx/default_logger.txt (train logger)
  - experiment/ppo_xxx/checkpoints (checkpoints dir)
  - experiment/ppo_xxx/data (actor generated data dir, if 'save_data=True')


You can use **sinfo** to inspect avaliable partitions.

Testing
---------

1. Configure checkpoint 

.. code-block:: yaml

    saver: # Required.
        save_dir: checkpoints
        resume_model: checkpoints/ckpt_e13.pth # checkpoint to test

2. Set **ROOT** path and **cfg** path in **test.sh**. By default, it should look like below

.. note:: 
    make sure you set `-e` option

.. code-block:: bash

    #!/bin/bash
    
    T=`date +%m%d%H%M`
    ROOT=../../
    cfg=faster-rcnn-R50-FPN-1x.yaml
    
    export PYTHONPATH=$ROOT:$PYTHONPATH
    
    g=$(($2<8?$2:8))
    srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:$g --ntasks-per-node=$g \
        --job-name=R50-FPN \
    python $ROOT/tools/train_val.py \
      -e \
      --config=$cfg \
      2>&1 | tee test.log.$T
    
3. Start testing

.. code-block:: bash
 
    # ./test.sh <PARTITION> <num_gpu> <cfg_path>
    ./test.sh Test 8 faster-rcnn-R50-FPN-1x.yaml
