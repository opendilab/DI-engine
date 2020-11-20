worker.learner
===================

base_learner
-----------------

BaseLearner
~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.learner.base_learner.BaseLearner
    :members: __init__, _init, _setup_hook, _setup_wrapper, time_wrapper, _setup_dataloader, _setup_agent, _setup_computation_graph, _setup_optimizer, _train, register_stats, register_hook, run, close, call_hook, info, save_checkpoint, launch


register_learner
~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: nervex.worker.learner.base_learner.register_learner

create_learner
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: nervex.worker.learner.base_learner.create_learner



learner_hook
----------------------

Hook
~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.learner.learner_hook.Hook
    :members: __init__

LearnerHook
~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.learner.learner_hook.LearnerHook
    :members: __init__

LrSchedulerHook
~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.learner.learner_hook.LrSchedulerHook
    :members: __init__, __call__

LoadCkptHook
~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.learner.learner_hook.LoadCkptHook
    :members: __init__, __call__

SaveCkptHook
~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.learner.learner_hook.SaveCkptHook
    :members: __init__, __call__


LogShowHook
~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.learner.learner_hook.LogShowHook
    :members: __init__, __call__

LogReduceHook
~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.learner.learner_hook.LogReduceHook
    :members: __init__, __call__

register_learner_hook
~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: nervex.worker.learner.learner_hook.register_learner_hook

build_learner_hook_by_cfg
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: nervex.worker.learner.learner_hook.build_learner_hook_by_cfg

merge_hooks
~~~~~~~~~~~~~
.. automodule:: nervex.worker.learner.learner_hook.merge_hooks



comm_learner
-----------------

BaseCommLearner
~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.learner.comm.base_comm_learner.BaseCommLearner
    :members: __init__, register_learner, send_agent, get_data, send_train_info, start_heartbeats_thread, init_service, close_service

FlaskFileSystemLearner
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.learner.comm.flask_fs_learner.FlaskFileSystemLearner
    :members: __init__, register_learner, send_agent, get_data, send_train_info, start_heartbeats_thread, init_service, close_service

LearnerCommHelper
~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.learner.comm.LearnerCommHelper
    :members: enable_comm_helper

add_comm_learner
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: nervex.worker.learner.comm.add_comm_learner

