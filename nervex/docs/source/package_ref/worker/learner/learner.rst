worker.learner
===================

base_learner
-----------------



BaseLearner
~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.learner.base_learner.BaseLearner
    :members: __init__, _setup_hook, _setup_wrapper, time_wrapper, _setup_data_source, _setup_computation_graph, _setup_optimizer, _get_data, _train, register_stats, run, close, call_hook, info, save_checkpoint


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

LrSchdulerHook
~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.learner.learner_hook.LrSchdulerHook
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

