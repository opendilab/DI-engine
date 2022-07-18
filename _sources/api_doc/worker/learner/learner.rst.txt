worker.learner
===================

learner_hook
----------------------

Please Reference ding/worker/learner/learner_hook.py for usage

Hook
~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.worker.learner.learner_hook.Hook
    :members: __init__

LearnerHook
~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.worker.learner.learner_hook.LearnerHook
    :members: __init__

LoadCkptHook
~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.worker.learner.learner_hook.LoadCkptHook
    :members: __init__, __call__

SaveCkptHook
~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.worker.learner.learner_hook.SaveCkptHook
    :members: __init__, __call__


LogShowHook
~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.worker.learner.learner_hook.LogShowHook
    :members: __init__, __call__

LogReduceHook
~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.worker.learner.learner_hook.LogReduceHook
    :members: __init__, __call__

register_learner_hook
~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: ding.worker.learner.learner_hook.register_learner_hook

build_learner_hook_by_cfg
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: ding.worker.learner.learner_hook.build_learner_hook_by_cfg

merge_hooks
~~~~~~~~~~~~~
.. automodule:: ding.worker.learner.learner_hook.merge_hooks


base_learner
-----------------

Please Reference ding/worker/learner/base_learner.py for usage

BaseLearner
~~~~~~~~~~~~~~~~~
.. autoclass:: ding.worker.learner.base_learner.BaseLearner
    :members: __init__, register_hook, train, start, setup_dataloader, close, call_hook, save_checkpoint, _setup_hook, _setup_wrapper

create_learner
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: ding.worker.learner.base_learner.create_learner


