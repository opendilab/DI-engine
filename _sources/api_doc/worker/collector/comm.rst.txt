
worker.collector.comm
========================

base_comm_collector
---------------------

Please Reference ding/worker/collector/comm/base_comm_collector.py for usage

BaseCommCollector
~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.worker.collector.comm.base_comm_collector.BaseCommCollector
    :members: __init__, get_policy_update_info, send_metadata, send_stepdata, start, close, _create_collector

create_comm_collector
~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: ding.worker.collector.comm.base_comm_collector.create_comm_collector



flask_fs_collector
---------------------

Please Reference ding/worker/collector/comm/flask_fs_collector.py for usage

CollectorSlave
~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.worker.collector.comm.flask_fs_collector.CollectorSlave
    :members: __init__, _process_task

FlaskFileSystemCollector
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.worker.collector.comm.flask_fs_collector.FlaskFileSystemCollector
    :members: __init__, deal_with_resource, deal_with_collector_start, deal_with_collector_data, deal_with_collector_close, get_policy_update_info, send_stepdata, send_metadata, start, close


utils
-----------------

Please Reference ding/worker/collector/comm/utils.py for usage

NaiveCollector
~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.worker.collector.comm.utils.NaiveCollector
    :members: _process_task, _get_timestep
