Unit Test Guide
=========================

Significance of Unit Testing
----------------------------------------

In the field of software engineering, unit testing is a testing method. Through this method, each unit of the source code set of one or more computer program modules and related control data, use programs and operation programs are tested to determine whether they can operate correctly (from `Wikipedia - Unit testing <https://en.wikipedia.org/wiki/Unit_testing>`_).

In actual development, the significance of unit testing is as follows:

* When the code is updated, you can run unit tests to ensure that regression errors do not occur.
* Through fine-grained unit test design, you can quickly and accurately locate the source of errors during unit test.
* Combining unit testing with code coverage ensures that all code and branches have been tested.
* After finding bugs, you can add test cases that can reproduce bugs to unit tests to continuously improve the perfection of code functions.
* Another important point -- **for a module, reading unit test code is also a very efficient way to understand its function and usage**.


.. _ref-test-types:

Types of Unit Test
---------------------------------

In the DI-engine project, we divide the unit test into the following parts:

* ``unittest`` -- functional unit test in a general sense to ensure the normal function of engineering code and the convergence of algorithm code on simple use cases.
* ``algotest`` -- unit test for algorithm code to ensure that the algorithm code can meet the use requirements on specific use cases.
* ``cudatest`` -- unit test for CUDA dependent features, ensure that such features function normally in the operating environment with CUDA.
* ``envpooltest`` -- unit test for features that rely on envpool high-performance parallel computing to ensure that such features function normally.
* ``platformtest`` -- unit test of cross platform code, ensure that the core functions of DI-engine can still operate normally on MacOS and Windows platforms.
* ``benchmark`` -- performance test of algorithm or architecture, speed measurement is mainly carried out for relevant contents to ensure that its performance meets the requirements.



How to Build Unit Test
---------------------------------

In DI-engine, we use `pytest <https://docs.pytest.org/>`_ to build unit tests.

For unit test writing, you can refer to the ``tests`` folder under the code path at all levels as a whole, such as `ding/envs/env_manager/tests <https://github.com/opendilab/DI-engine/tree/main/ding/envs/env_manager/tests>`_.


Naming Conventions
~~~~~~~~~~~~~~~~~~~~~~~~

For unit testing, we generally build it in the unit of class or function, and its name should meet certain specifications, specifically:

* For the unit test of function form, the function name is required to be started with ``test_``.
* For the unit test of class form, the class name is required to be started with ``Test``, and names of all methods used for testing should start with ``test_``.


Assertions
~~~~~~~~~~~~~~~~~~~~~~~~

In the test cases, we use ``assert`` (assertion) to check the prototype results. If the assertion is not true, very detailed information will be displayed, as shown in the following figure

.. image:: pytest_assert.png
    :scale: 55%
    :align: center

In addition, `` pytest`` also supports assertion of thrown exceptions, as shown below

.. code:: python

   import pytest

   @pytest.mark.unittest
   def test_zero_division():
       with pytest.raises(ZeroDivisionError):
           1 / 0

In addition, for the test of real numbers, due to the storage principle of real numbers, it may lead to misjudgment caused by subtle errors. Therefore, the approximate function ``approx`` can be used for approximate judgment. It supports numeric type, ``list`` type, ``dict`` type and ``numpy`` type (``numpy. Ndarray``).

.. image:: pytest_approx.png
    :scale: 55%
    :align: center



Fixture and Conftest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fixture is a very important mechanism in ``pytest``. It can initialize the resources required for testing and pass them in as parameters of the test function for its usage. Not only that, but also the recovery of operation resources can be realized to ensure that the subsequent operation will not be affected. In addition, code reuse can be easily realized through the definition of scope.

This `fixture tutorial <https://www.lambdatest.com/blog/end-to-end-tutorial-for-pytest-fixtures-with-examples/>`_ is written in great detail and can be used as a reference. In the existing code of DI-engine, you can refer to `ding/league/tests/test_player.py <https://github.com/opendilab/DI-engine/tree/main/ding/league/tests/test_player.py>`_.

Fixture is generally used in a single file, that is, it is used after defining fixture under the current file. If you need to use fixture across files, you can use the ``conftest`` (abbreviation of ``config of test``) mechanism. There is no need to explicitly import in the test file, and the ``pytest`` framework will automatically complete the loading. You can refer to this `tutorial <https://www.lambdatest.com/blog/end-to-end-tutorial-for-pytest-fixtures-with-examples/#Sharingpytest>`_, and in the existing code, you can refer to `ding/league/tests/conftest.py <https://github.com/opendilab/DI-engine/tree/main/ding/league/tests/conftest.py>`_.


Test Mark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to distinguish the types of tests (for example, ref:`ref-test-types`), you can add ``pytest.mark("MARK-NAME")`` decorator to let the test be executed by category, and use ``pytest –m MARK-NAME`` to execute the selected type of test at run time.

.. image:: pytest_mark.png
    :scale: 55%
    :align: center


Parameterized
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In some cases, we need to reuse the same test logic and test for different input data. At this point, we can use parameter configuration ``@pytest.mark.paramtrize(argsnames, argsvalues, ids=None)`` realize parameter configuration for multiple groups of tests. Of which:

-  ``argsnames``
   : means parameter name, with type of ``str``. If you need to express multiple parameter names, use commas to separate them.

-  ``argsvalues``
   : means parameter value, with type if ``list`` which is composed of parameters. The elements in the list are the values assigned to the parameters. If multiple parameters are set in ``argsnames``, the ``tuple`` type will be used, and the values will correspond to the names one by one in order.

For example:

- If using decorator ``@pytest.mark.paramtrize('data', [1, 2, 3])``, then the `` data`` variable will be assigned to 1, 2 and 3 respectively for test.
- If using decorator ``@pytest.mark.paramtrize('var1, var2', [(1, 2), (2, 3), (3, 4)])``, the ``(var1, var2)`` variables will be assigned ``(1, 2)``, ``(2, 3)``, ``(3, 4)`` test.

You can refer to the writing method in `ding/utils/data/tests/test_dataloader.py <https://github.com/opendilab/DI-engine/tree/main/ding/utils/data/tests/test_dataloader.py>`_.



How do Run Unit Test
---------------------------------

In DI-engine, we use ``pytest`` to start unit tests. For very simple cases, you can use the command directly:

.. code-block:: shell

   pytest -sv ./ding

When you need to know the unit test coverage and specific coverage distribution, you need to use the following commands:

.. code-block:: shell

   pytest -sv ./ding -m unittest --cov-report term-missing --cov=./ding

The meanings of each parameter are as follows:

- ``-m`` ： Select the type of marks to test.
- ``-s`` ： The output content is not captured, which is the abbreviation of ``--capture=no`` option.
- ``-v`` ： Select the complexity level of the output content. The currently selected is a lower complexity level. If you need to output more detailed information, you can use ``-vv`` to increase the complexity, and so on.
- ``--cov-report term-missing`` ： Select to display the coverage report in the form of ``term-missing``, which refers to "display the specific areas not covered".
- ``--cov`` ： Select the code area to be overwritten.

.. note::

   A more recommended method is to use the encapsulated script in the ``Makefile`` for quick startup, for example:

   .. code-block:: shell

      make unittest  # Full unit testing
      make unittest RANGR_DIR=./ding/xxx  # Test for specific sub modules
      make algotest
      make cudatest
      make envpooltest
      make platformtext



