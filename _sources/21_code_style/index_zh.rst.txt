代码风格指南
=======================

代码风格
-------------------

在 `DI-engine <https://github.com/opendilab/DI-engine>`_ 中，我们遵循如下的基本代码规范：

* 对于 **文件名**，一律使用小写字母、数字与下划线进行命名，例如 ``my_policy.py`` 。
* 对于 **类** （Class），一概采用大写字母开头的 `驼峰命名法 <https://en.wikipedia.org/wiki/Camel_case>`_ 进行命名，例如 ``MyClass`` ；另外，对于内部类可以在开头使用额外的下划线，例如： ``_InnerClass`` 。
* 对于 **函数** （Function）和 **方法** （Method），一律使用小写字母、数字与下划线进行命名，例如 ``my_function`` 、 ``my_method`` 。
* 对于 **变量** （Variable），一律使用小写字母、数字与下划线进行命名，例如 ``my_var`` 。
* 对于归属于类的方法与变量，使用单个下划线表述受保护的继承关系，例如 ``_protected_val`` ；使用两个下划线表述私有的继承关系，例如 ``__private_val`` 。
* 对于方法参数的命名，若为实例方法则第一个参数应命名为 ``self`` ，若为类方法则第一个参数应命名为 ``cls`` ；列表变长参数请使用 ``*args`` ，键值对参数使用 ``**kwargs`` 。
* 变量命名时，如果名称与保留的关键字、原生的类等发生了冲突，请在末尾加上下划线以避免造成非预期的影响，例如 ``type_`` 。



yapf
-------------------

对于 `yapf <https://github.com/google/yapf>`_ ，我们可以使用现有的 `Makefile <https://github.com/opendilab/DI-engine/blob/main/Makefile>`_ 进行一键修复

.. code-block:: shell

   make format


考虑到整个项目规模较大，文件数量较多，因此可以使用下列命令对特定路径下的源代码文件进行代码风格一键修复

.. code-block:: shell

   make format RANGE_DIR=./ding/xxx


在该项目中，我们使用基于PEP8的 `yapf代码规范配置 <https://github.com/opendilab/DI-engine/blob/main/.style.yapf>`_ ，关于配置的详细信息，可以参考 `Github主页的描述 <https://github.com/google/yapf#knobs>`_ 。 `PEP8 <https://peps.python.org/pep-0008/>`_ 为Python官方推荐的代码风格配置，对代码风格的注重可以提高代码的可读性，也可以最大限度减少不符合预期的行为。

此外，yapf还可以通过插件yapf-pycharm与PyCharm进行集成：

* `yapf-pycharm <https://plugins.jetbrains.com/plugin/9705-yapf-pycharm>`_


flake8
-------------------

对于 `flake8 <https://github.com/PyCQA/flake8>`_ ，我们可以使用现有的 `Makefile <https://github.com/opendilab/DI-engine/blob/main/Makefile>`_ 进行代码设计上的检查

.. code-block:: shell

   make flake_check


考虑到整个项目规模较大，文件数量较多，因此可以使用下列命令对特定路径下的源代码文件进行代码设计上的检查

.. code-block:: shell

   make flake_check RANGE_DIR=./ding/xxx


在该项目中，我们使用基于PEP8的 `flake8代码设计规范配置 <https://github.com/opendilab/DI-engine/blob/main/.flake8>`_ ，关于配置的详细信息，可以参考 `flake8官方文档的描述 <https://flake8.pycqa.org/en/latest/user/configuration.html>`_ 。 `PEP8 <https://peps.python.org/pep-0008/>`_ 为Python官方推荐的代码风格配置，对代码风格的注重可以提高代码的可读性，也可以最大限度减少不符合预期的行为。



