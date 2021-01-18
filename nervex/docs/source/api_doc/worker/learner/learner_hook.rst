worker.learner
===================

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

