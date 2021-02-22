
worker.actor.comm
========================

base_comm_actor
-----------------

Please Reference nervex/worker/actor/comm/base_comm_actor.py for usage

BaseCommActor
~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.actor.comm.base_comm_actor.BaseCommActor
    :members: __init__, get_policy_update_info, send_metadata, send_stepdata, start, close, _create_actor

register_comm_actor
~~~~~~~~~~~~~~~~~~~~~
.. automodule:: nervex.worker.actor.comm.base_comm_actor.register_comm_actor

create_comm_actor
~~~~~~~~~~~~~~~~~~~
.. automodule:: nervex.worker.actor.comm.base_comm_actor.create_comm_actor



flask_fs_actor
-----------------

Please Reference nervex/worker/actor/comm/flask_fs_actor.py for usage

ActorSlave
~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.actor.comm.flask_fs_actor.ActorSlave
    :members: __init__, _process_task

FlaskFileSystemActor
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.actor.comm.flask_fs_actor.FlaskFileSystemActor
    :members: __init__, deal_with_resource, deal_with_actor_start, deal_with_actor_data, deal_with_actor_close, get_policy_update_info, send_stepdata, send_metadata, start, close


utils
-----------------

Please Reference nervex/worker/actor/comm/utils.py for usage

NaiveActor
~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.actor.comm.utils.NaiveActor
    :members: _process_task, _get_timestep
