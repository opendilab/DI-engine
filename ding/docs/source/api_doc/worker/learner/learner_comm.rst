worker.learner.comm
===================


learner_comm
-----------------

BaseCommLearner
~~~~~~~~~~~~~~~~~
.. autoclass:: ding.worker.learner.comm.base_comm_learner.BaseCommLearner
    :members: __init__, send_policy, get_data, send_learn_info, start, close

FlaskFileSystemLearner
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.worker.learner.comm.flask_fs_learner.FlaskFileSystemLearner
    :members: __init__, send_policy, get_data, send_learn_info, start, close

create_comm_learner
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: ding.worker.learner.comm.base_comm_learner.create_comm_learner

