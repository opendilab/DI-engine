interaction.master.connection
==============================

_ISlaveConnection
~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.interaction.master.connection._ISlaveConnection
    :members: connect, disconnect, new_task, start, close


SlaveConnection
~~~~~~~~~~~~~~~~~

.. autoclass:: ding.interaction.master.connection.SlaveConnection
    :members: __init__, is_connected, _before_connect, _after_connect, _error_connect, _before_disconnect, _after_disconnect, _error_disconnect, _before_new_task, _after_new_task, _error_new_task, connect, disconnect, new_task, start, close

SlaveConnectionProxy
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.interaction.master.connection.SlaveConnectionProxy
    :members: __init__, is_connected, connect, disconnect, new_task, start, close

