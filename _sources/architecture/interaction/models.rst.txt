Interaction Model
====================

The uml structure of the main models of the interaction framework is like below:

.. image:: model.puml.svg
   :align: center

(UML diagram powered by plantuml_. Automatic generation of uml diagrams powered by plantumlcli_.)

* After `new_connection` method called in master client, a new connection object will be created (but not actually connected to slave yet).
* After `connect` method called in connection object, the master will try to connect to the given slave client. If success, the connection will be actually established, otherwise this connection is failed.
* After the connection is established, new task can be sent to slave end by calling `new_task` method.
* In the end, `disconnect` can be called in connection object, the master will disconnect from the slave end.


.. _plantuml: https://plantuml.com/
.. _plantumlcli: https://github.com/HansBug/plantumlcli

