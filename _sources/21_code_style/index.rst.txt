Code Style Guide
==========================

Basic Code Style Rules
---------------------------

In `DI-engine <https://github.com/opendilab/DI-engine>`_, we follow the following basic code specifications:

* The **file name** is always named with lowercase letters, numbers and underscores, such as ``my_policy.py``.
* For **class**, the `Camelcase Naming Convention <https://en.wikipedia.org/wiki/Camel_case>`_ which is started with capital letters shall be adopted, such as ``MyClass``. In addition, for internal classes, you can use additional underscores at the beginning, for example ``_InnerClass``.
* All **functions** and **methods** are named with lowercase letters, numbers and underscores, such as ``my_function`` and ``my_method``.
* For **variables**, they are all named with lowercase letters, numbers and underscores, such as ``my_var``.
* For methods and variables belonging to a class, use a single underscore to express the protected inheritance relationship, such as ``_protected_val``. Use two underscores to express a private inheritance relationship, for example ``__private_val``.
* For the naming of method parameters, if it is an instance method, the first parameter should be named ``self``. If it is a class rule, the first parameter should be named ``cls``. Please use ``*args`` for the variable length parameter of the list, and ``**kwargs`` for the key value parameter.
* When naming variables, if the name conflicts with reserved keywords, native classes, etc., please underline at the end to avoid unexpected effects, such as ``type_``.



yapf
-------------------

For `yapf <https://github.com/google/yapf>`_, we can use existing `Makefile <https://github.com/opendilab/DI-engine/blob/main/Makefile>`_ for one-click fix

.. code-block:: shell

   make format


Considering the large scale of the whole project and the large number of files, you can use the following commands to check the code design of the source code files in a specific path

.. code-block:: shell

   make format RANGE_DIR=./ding/xxx


In this project, we use the `yapf config <https://github.com/opendilab/DI-engine/blob/main/.style.yapf>`_ code specification configuration based on PEP8. For details about the configuration, you can refer to `the description on the Github homepage <https://github.com/google/yapf#knobs>`_. `PEP8 <https://peps.python.org/pep-0008/>`_ is the code style configuration officially recommended by Python. Paying attention to code style can improve the readability of the code and minimize unintended behavior.

In addition, yapf can also integrate with pycharm through the plug-in yapf pycharm:

* `yapf-pycharm <https://plugins.jetbrains.com/plugin/9705-yapf-pycharm>`_


flake8
-------------------

For `flake8 <https://github.com/PyCQA/flake8>`_, we can use the existing `Makefile <https://github.com/opendilab/DI-engine/blob/main/Makefile>`_ to check the code design

.. code-block:: shell

   make flake_check


Considering the large scale of the whole project and the large number of files, you can use the following commands to check the code design of the source code files in a specific path

.. code-block:: shell

   make flake_check RANGE_DIR=./ding/xxx


In this project, we use `flake8 code design specification configuration <https://github.com/opendilab/DI-engine/blob/main/.flake8>`_ based on pep8. For details of configuration, please refer to `the description of flake8 official documents <https://flake8.pycqa.org/en/latest/user/configuration.html>`_. `PEP8 <https://peps.python.org/pep-0008/>`_ is the code style configuration officially recommended by python. Paying attention to the code style can improve the readability of the code and minimize the behavior that does not meet the expectations.



