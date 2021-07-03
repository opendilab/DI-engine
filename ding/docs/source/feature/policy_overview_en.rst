Policy Overview
===================


Policy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Policy Modes:

    1. 3 Common Mode
        
        * ``learn_mode`` : a collection of functions/methods which are designed to update and optimize policy.

            Here is a demo about how to apply policy ``learn_mode`` in serial pipeline:

            .. image:: images/serial_learner.png
                :scale: 60%

        * ``collect_mode`` : a collection of functions/methods which aims to collect training data with the balance of exploration and exploitation.

            Here is a demo about how to apply policy ``collect_mode`` in serial pipeline:

            .. image:: images/serial_collector.png
                :scale: 60%

        * ``eval_mode`` : a collection of functions/methods which are responsible for fair policy evaluation.

            Here is a demo about how to apply policy ``eval_mode`` in serial pipeline:

            .. image:: images/serial_evaluator.png
                :scale: 60%

    2. Some Customrized Mode(User defined)

        * ``command_mode`` : a collection of functions/methods for information control among different modes.

        * ``league mode`` : a collection of functions/methods related to self-play league training.

        * ``trick mode`` : a collection of functions/methods for adaptive hyper-parameter tuning.

Policy Interfaces:

    1. Common Interfaces:

        * ``default_config``

        * ``__init__``

        * ``_create_model``

        * ``_set_attribute``

        * ``_get_attribute``

        * ``sync_gradients``

        * ``default_model``

    2. Learn Mode Interfaces:

        * ``_forward_learn``

        * ``_reset_learn``

        * ``_monitor_vars_learn``

        * ``_state_dict_learn``

        * ``_load_state_dict_learn``

    3. Collect Mode Interfaces:

        * ``_forward_collect``

        * ``_reset_collect``

        * ``_process_transition``

        * ``_get_train_sample``

        * ``_state_dict_collect``

        * ``_load_state_dict_collect``

    4. Eval Mode Interfaces:

        * ``_forward_eval``

        * ``_reset_eval``

        * ``_state_dict_eval``

        * ``_load_state_dict_eval``


All the mentioned above are some basic definitions and instructions, the users can learn from the examples.(ding/policy/)

.. note::
   How to define own get_train_sample case

.. note::
   How to define policy config

.. note::
   How to customize model in different modes

.. tip::
   Many algorithms use target model to surpass over estimation, please be careful to related variable in policy.
