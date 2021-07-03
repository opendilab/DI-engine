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

   Python version: 3.6, 3.7, 3.8 (You can refer to `Python Installation Guide <https://pytorch.org/get-started/locally/#linux-python>`_ in PyTorch doc. Please pay attention to the correct Python version.) 

   PyTorch version: >=1.3.1, <=1.7.1 (You can use any proper version in this range, see `PyTorch Installation <https://pytorch.org/get-started/locally/>`_)

   .. note::

        If there is a GPU in your setting, PyTorch with CUDA runtime is recommended. Otherwise, you just need to install cpu version PyTorch.


Stable Release Version
**********************************************************

You can simply install DI-engine from PyPI with the following command:

.. code-block:: bash

     pip install DI-engine

.. tip::

    If you encounter timeout in downloading packages, you can try to indicate the corresponding pip source according to your area.
    

And if you prefer to use Anaconda or Miniconda, the next command is suggested:

.. code-block:: bash

    conda -c conda-forge install DI-engine

Also, you can install DI-engine from the source codes in github(master branch recommended)

.. code-block:: bash

    git clone https://gitlab.bj.sensetime.com/open-XLab/cell/ding.git
    cd DI-engine
    pip install . --user

If you want to install the extra package required by some functions in DI-engine(such as concrete env, unittest and doc), you can execute

.. code-block:: bash

     pip install DI-engine[common_env]  # install atari-env and box-2d env
     pip install DI-engine[test]  # install unittest(pytest) related package

If you complete installation with the similar output in your terminal, the installation is over gracefully and you can check it with the next section.

.. code-block:: bash

    Installing collected packages: DI-engine
      Running setup.py develop for DI-engine
    Successfully installed DI-engine

.. tip::
    Some shells such as Zsh require quotation marks around package names, i.e. pip install 'DI-engine[test]' 


.. note::
    
   The whole installation procedure often lasts about 30 seconds(depends on the download speed of packages), if there are some failed packages, you can also refer to ``setup.py`` and install the specific
   package manually.

Development Version
********************
To contribute to DI-engine, with support for running tests and building the documentation, you need to find proper release tag or master branch.

.. code-block:: bash

    # source r0.3.2  # maybe you need activate virtual environment first

    git clone http://gitlab.bj.sensetime.com/open-XLab/cell/ding.git && cd DI-engine
    pip install -e .[doc, test, common_env] --user

Check install
****************

After installation, you can open your python console and run the following codes

.. code-block:: python

    import DI-engine
    print(ding.__version__)

If the console print the correct version tag, you have successfully installed DI-engine

Besides, DI-engine also prepare the CLI tool for users, you can type the following command in your terminal

.. code-block:: bash

   DI-engine -v

If the terminal return the correct information, you can use this CLI tool for the common training and evaluation, you can type ``DI-engine -h`` for further usage
