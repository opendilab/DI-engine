worker.learner
===================


learner_comm
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

