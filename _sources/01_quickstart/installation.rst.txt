Installation Guide
============================

.. toctree::
   :maxdepth: 2

Prerequisites
--------------

Operation system version: Linux, macOS, Windows

Python version: 3.6-3.8 

.. note::

    If there is a GPU in your setting, you can refer to `Nvidia CUDA Toolkit Installation <https://developer.nvidia.com/cuda-downloads/>`_
    
    After CUDA being installed, you will get a correct Nvidia CUDA version of Pytorch automaticly when installing DI-engine.
    
    If you want to install Pytorch manually, you can refer to `PyTorch Installation <https://pytorch.org/get-started/locally/>`_.

    If your OS is Windows, please do confirm that SWIG is installed and available through the OS environment variable PATH, you can refer to `SWIG installation <https://www.swig.org/download.html>`_.

Stable Release Version
------------------------

You can simply install stable release DI-engine with the following command:

.. code-block:: bash

    # Current stable release of DI-engine
    pip install DI-engine

.. tip::

    If you need to upgrade pip, you can use the following commands:

    .. code-block:: bash

        # Windows
        > python -m pip install --upgrade pip
        # Linux
        $ pip install --upgrade pip
        # MacOS
        $ pip install --upgrade pip

.. tip::

    If you encounter timeout in downloading packages, you can try to request from other site.

    .. code-block:: bash

        pip install requests -i https://mirrors.aliyun.com/pypi/simple/ DI-engine    

And if you prefer to use Anaconda or Miniconda, the following command is suggested:

.. code-block:: bash

    conda install -c opendilab di-engine

Development Version
----------------------

If you need to install latest DI-engine in development from the Github source codes:

.. code-block:: bash

    git clone https://github.com/opendilab/DI-engine.git
    cd DI-engine
    pip install .

.. tip::

    If you hope to install DI-engine into local user directories, you can do as the following:

    .. code-block:: bash

        pip install . --user

    Be careful that if you are using virtual python environment created by softwares, such as virtualenv, then the option "--user" may not work. Please ignore this tip.

Special Version
-------------------

If you want to enable special version of DI-engine and install the extra packages that are required, you can use the following command:

.. code-block:: bash

    # install atari and box-2d related packages
    pip install DI-engine[common_env]
    # install unittest(pytest) related packages
    pip install DI-engine[test]
    # enable numba acceleration
    pip install DI-engine[fast]
    # install multi extra packages
    pip install DI-engine[common_env,test,fast]

.. tip::

    Some certain shells require quotation marks around package names as the folloing:

    .. code-block:: bash

        pip install 'DI-engine[common_env,test,fast]'

.. note::

    The whole installation procedure often lasts about 30 seconds, which depends on the the size of packages as well as download speed. 
    If some packages installation failed, you can refer to the file "setup.py" and install the specific package manually.

Run in Docker
--------------

DI-engine docker images are available in `DockerHub <https://hub.docker.com/r/opendilab/ding>`_. You can use the following commands to pull the image:

.. code-block:: bash

    # Download Stable release DI-engine Docker image
    docker pull opendilab/ding:nightly 
    # Run Docker image
    docker run -it opendilab/ding:nightly /bin/bash

Installation Check
-------------------

After installation, you can use the following python codes to check if DI-engine is available and show the version of it:

.. code-block:: python

    import ding
    print(ding.__version__)

You can also try the command line tool of DI-engine as the folloing:

.. code-block:: bash

    ding -v

