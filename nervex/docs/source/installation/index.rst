Installation
===============================

.. toctree::
   :maxdepth: 3

Prerequisites
***********************************

   System version:

     - Ubuntu 16.04, 18.04, 20.04
     - Centos Linux version 3.10.0-693.el7.x86_64
     - macOS
     - Windows10

   Python version: 3.6, 3.7, 3.8     


Stable Release Version
**********************************************************

You can simply install nerveX from PyPI with the following command:

.. code-block:: bash

     pip install nervex

If you want to install the extra package required by some functions in nerveX(such as concrete env, unittest and doc), you can execute

.. code-block:: bash

     pip install nervex[common_env]  # install atari-env and box-2d env
     pip install nervex[test]  # install unittest(pytest) related package

.. tip::
    Some shells such as Zsh require quotation marks around package names, i.e. pip install 'nervex[test]' 


Development Version
********************
To contribute to nerveX, with support for running tests and building the documentation, you need to find proper release tag or master branch.

.. code-block:: bash

    # source r0.3.2  # maybe you need activate virtual environment first

    git clone http://gitlab.bj.sensetime.com/open-XLab/cell/nerveX.git && cd nerveX
    pip install -e .[doc, test, common_env] --user

Check install
****************

After installation, you can open your python console and run the following codes

.. code-block:: python

    import nervex
    print(nervex.__version__)

If the console print the correct version tag, you have successfully installed nerveX

Besides, nerveX also prepare the CLI tool for users, you can type the following command in your terminal

.. code-block:: bash

   nervex -v

If the terminal return the correct information, you can use this CLI tool for the common training and evaluation, you can type ``nervex -h`` for further usage
