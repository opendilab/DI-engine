Diagrams and Visualization
========================================

In DI-engine, we often need to draw images and measure and visualize some information. This section will introduce these contents in detail.


PlantUML
-----------------

PlantUML is a tool that can be used to draw UML and other images. For details, please refer to `the official website of PlantUML <https://plantuml.com/zh/>`_. Its biggest feature is that it is based on code, does not need to pay attention to typesetting, and is very easy to maintain.

For example, we can draw class diagrams

.. image:: plantuml-class-demo.puml.svg
    :align: center

You can draw the flow chart of the algorithm

.. image:: plantuml-activity-en-demo.puml.svg
    :align: center

YAML data can also be plotted

.. image:: plantuml-yaml-demo.puml.svg
    :align: center

We can use plantumlcli tool to generate images. For details, please refer to `plantumlcli GitHub repository <https://github.com/HansBug/plantumlcli>`_.

.. note::

    In the document of DI-engine, plantuml has been integrated, which can automatically generate images based on source code. For example, we can create the file ``plantuml-demo.puml`` under the current path.

    .. literalinclude:: plantuml-demo.puml
        :language: text
        :linenos:

    When compiling the document, the image ``plantuml-demo.puml.svg`` in SVG format will also be generated automatically, as shown below.

    .. image:: plantuml-demo.puml.svg
        :align: center



graphviz
-----------------

For more complex topology diagrams, we can use tool Graphviz to draw:

* `Official Documentation of Graphviz <https://graphviz.org/>`_
* `Python Wrapper Library of Graphviz  <https://github.com/xflr6/graphviz>`_
* `Graphviz Online <https://dreampuf.github.io/GraphvizOnline/>`_

For example, we can use graphviz to quickly draw a graph structure, as shown in the following code

.. literalinclude:: graphviz-demo.gv
    :language: text
    :linenos:

The drawn image is shown below

.. image:: graphviz-demo.svg
    :align: center



draw.io
-----------------

``draw.io`` is a very simple and easy-to-use online image editing tool, which can be used to edit workflow diagram, BPM, organization diagram, UML diagram, ER diagram, network topology, etc:

* `Drawing Tool of draw.io <https://www.draw.io/>`_
* `Official Site of draw.io <https://drawio-app.com/>`_

.. image:: draw.io-example.png
    :align: center

``draw.io``'s biggest feature is the drag drawing method, so it can realize "what you see is what you get".



snakeviz
-----------------

When we need to measure the speed of a program or part of a program, we can use the native ``cProfile``, while ``sneakviz`` can display the speed measurement results in a visual form:

* `Official Documentation of cProfile <https://docs.python.org/3/library/profile.html>`_
* `Official Documentation of snakeviz <https://jiffyclub.github.io/snakeviz/#:~:text=SnakeViz%20is%20a%20browser%20based,Python%202.7%20and%20Python%203.>`_

.. image:: snakeviz-example.png
    :align: center



