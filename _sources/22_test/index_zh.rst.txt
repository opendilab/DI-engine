单元测试指南
==============================

单元测试的意义
---------------------

在软件工程领域，单元测试是一种测试方法，通过这种方法，对一个或多个计算机程序模块的源代码集的各个单元以及相关的控制数据、使用程序和操作程序进行测试，以确定它们是否能正确运行（摘自 `维基百科 - Unit testing <https://en.wikipedia.org/wiki/Unit_testing>`_ ）。

在实际开发中，单元测试的意义如下：

* 在代码被更新时，可以通过运行单元测试来确保不会出现回归错误。
* 通过细粒度的单元测试设计，可以在单元测试时快速且精确地定位到错误源头。
* 将单元测试与代码覆盖率结合，可以确保所有代码和分支均受到过测试。
* 在发现了Bug后，可以将可复现Bug的测试用例添加至单元测试，让代码功能的完善性不断提高。
* 另外很重要的一点——**对于一个模块而言，想要了解其功能和使用方式，阅读单元测试代码也是非常高效的一种手段**。


.. _ref-test-types-zh:

测试类型
---------------------

在DI-engine项目中，我们将单元测试分为如下的若干部分：

* ``unittest`` ——为一般意义上的功能性单元测试，确保工程代码功能正常，算法代码在简单用例上能实现收敛。
* ``algotest`` ——为针对算法代码的单元测试，确保算法代码在特定的用例上能满足使用需求。
* ``cudatest`` ——为针对依赖于CUDA的特性的单元测试，确保此类特性在有CUDA的运行环境上功能正常。
* ``envpooltest`` ——为针对依赖于envpool高性能并行计算的特性的单元测试，确保此类特性功能正常。
* ``platformtest`` ——为针对跨平台代码的单元测试，确保DI-engine的核心功能在MacOS和Windows平台上依然可以正常运行。
* ``benchmark`` ——为针对算法或架构的性能测试，主要针对相关内容进行测速，确保其性能满足要求。




如何编写单元测试
---------------------

在DI-engine中，我们使用 `pytest <https://docs.pytest.org/>`_ 进行单元测试的搭建。

对于单元测试的撰写，整体上可以参考各级代码路径下的 ``tests`` 文件夹，例如 `ding/envs/env_manager/tests <https://github.com/opendilab/DI-engine/tree/main/ding/envs/env_manager/tests>`_ 。


命名规范
~~~~~~~~~~~~~~~~~~~

对于单元测试，我们一般以类或函数为单位进行搭建，其命名应当满足一定的规范，具体为：

* 对于函数形态的单元测试，要求函数以 ``test_`` 开头。
* 对于类形态的单元测试，要求类名以 ``Test`` 开头，并且各个用于测试的方法均以 ``test_`` 开头。


断言
~~~~~~~~~~~~~~~~~~~

在测试用例中，我们使用 ``assert`` （断言）对原型结果进行检查。如果断言不成立，则会显示非常详细的信息，如下图所示

.. image:: pytest_assert.png
    :scale: 55%
    :align: center

不仅如此， ``pytest`` 还支持对抛出的异常进行断言，如下所示

.. code:: python

   import pytest

   @pytest.mark.unittest
   def test_zero_division():
       with pytest.raises(ZeroDivisionError):
           1 / 0

另外，对于实数的测试，由于实数本身的存储原理，可能导致因为细微的误差造成的误判。因此可以使用近似函数 ``approx`` 进行近似判断，其支持数值类型、列表类型（ ``list`` ）、字典类型（ ``dict`` ）与numpy类型（ ``numpy.ndarray`` ）。

.. image:: pytest_approx.png
    :scale: 55%
    :align: center


固件与配置
~~~~~~~~~~~~~~~~~~~~

固件（fixture）是 ``pytest`` 中非常重要的机制，其可以完成测试所需资源的初始化，并作为测试函数的参数传入，供测试函数使用。不仅如此，还可以实现对运行资源的回收，确保后续运行不受影响。此外，还可以通过对作用域的定义，轻松地实现代码复用。

这篇 `fixture 中文教程 <https://www.cnblogs.com/linuxchao/p/linuxchao-pytest-fixture.html>`_ 写的很详细，可以作为参考。在DI-engine的现有代码中，可以参考 `ding/league/tests/test_player.py <https://github.com/opendilab/DI-engine/tree/main/ding/league/tests/test_player.py>`_ 。

固件一般在单个文件中使用，即在当前文件下定义固件后使用。如果需要跨文件使用固件，可以使用测试配置（conftest，config of test的缩写）机制实现。在测试文件中不需要显式地进行导入， ``pytest`` 框架会自动完成加载。可以参考这篇 `中文教程 <https://www.cnblogs.com/linuxchao/p/linuxchao-pytest-conftest.html>`_，在现有的代码中可以参考 `ding/league/tests/conftest.py <https://github.com/opendilab/DI-engine/tree/main/ding/league/tests/conftest.py>`_ 。



测试标记
~~~~~~~~~~~~~~~~~~~~~

为了对测试的类型进行区分（如:ref:`ref-test-types-zh`），可以添加 ``pytest.mark("MARK-NAME")`` 装饰器来让测试分类执行，并在运行时使用 ``pytest –m MARK-NAME`` 来执行所选择类型的测试。

.. image:: pytest_mark.png
    :scale: 55%
    :align: center


参数配置
~~~~~~~~~~~~~~~~~~~~~

部分情况下，我们需要复用同一段测试逻辑，针对不同的输入数据展开测试。此时我们可以使用参数配置（parameterize） ``@pytest.mark.paramtrize(argsnames, argsvalues, ids=None)`` 实现对多组测试的参数配置。其中：

-  ``argsnames``
   ：意为参数名，类型为字符串（ ``str`` ），如果需要表达多个参数名，则使用英文逗号进行分隔。

-  ``argsvalues``
   ：意为参数值，类型为由参数组成的列表（ ``list`` ），列表中的元素即为对参数赋的值，如果在 ``argsnames`` 中设置了多个参数，则使用元组（ ``tuple`` ）类型，并将值将与名字按照顺序一一对应。

例如：

* 若使用装饰器 ``@pytest.mark.paramtrize('data', [1, 2, 3])`` ，则会为 ``data`` 变量分别赋值为1、2、3进行测试
* 若使用装饰器 ``@pytest.mark.paramtrize('var1, var2', [(1, 2), (2, 3), (3, 4)])`` ，则会为 ``(var1, var2)`` 变量分别赋值为 ``(1, 2)`` 、 ``(2, 3)`` 、 ``(3, 4)`` 进行测试。

可以参考 `ding/utils/data/tests/test_dataloader.py <https://github.com/opendilab/DI-engine/tree/main/ding/utils/data/tests/test_dataloader.py>`_ 中的写法。



如何进行单元测试
---------------------

在DI-engine中，我们使用 ``pytest`` 启动单元测试。对于极为简单的情况，可以直接使用命令

.. code-block:: shell

   pytest -sv ./ding

当需要得知单元测试覆盖率及具体覆盖分布情况时，则需要用到如下命令：

.. code-block:: shell

   pytest -sv ./ding -m unittest --cov-report term-missing --cov=./ding

其中各个参数含义如下：

- ``-m`` ： 选择进行测试的标记类型。
- ``-s`` ： 不进行输出内容捕捉，为 ``--capture=no`` 的缩写形式。
- ``-v`` ： 选择输出内容的复杂级别，当前选择的为较低的复杂程度。如果需要输出更加详细的信息，可以使用 ``-vv`` 来增加复杂度，以此类推。
- ``--cov-report term-missing`` ： 选择以 ``term-missing`` 形式展示覆盖率报告，此处指“显示未覆盖的具体区域”。
- ``--cov`` ： 选择需要进行覆盖的代码区域。

.. note::

   一种更加推荐的做法是使用 ``Makefile`` 中封装完毕的脚本进行快速启动，例如：

   .. code-block:: shell

      make unittest  # 全面进行单元测试
      make unittest RANGR_DIR=./ding/xxx  # 针对特定子模块进行测试
      make algotest
      make cudatest
      make envpooltest
      make platformtext


