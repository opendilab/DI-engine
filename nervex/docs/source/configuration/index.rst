Configuration
==============

.. toctree::
   :maxdepth: 2


Learner Config
~~~~~~~~~~~~~~~
.. code:: yaml

    data:
        train:
            batch_size: 128
            dataloader_type: 'online'  # refer to data/online/online_dataloader
    train:
        use_cuda: True  
        use_distributed: True  # use multi-GPU training
        max_iterations: 1e9
        batch_size: 128
        trajectory_len: 16
    logger:
        print_freq: 5
        save_freq: 200
        eval_freq: 1000000
        var_record_type: 'alphastar'  # refer to log_helper
