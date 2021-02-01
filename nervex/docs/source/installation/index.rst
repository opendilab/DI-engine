Installation
===============================

.. toctree::
   :maxdepth: 3

1. Check system and python version
***********************************

   system version:

     - Ubuntu 16.04, 18.04, 20.04
     - Centos Linux version 3.10.0-693.el7.x86_64 (builder@kbuilder.dev.centos.org)
     - macOS
     - Windows10

   python version: 3.6, 3.7, 3.8     


2. Look up tags and find corresponding historical version 
**********************************************************

   release tag or master branch recommended

.. code-block:: bash

     git clone http://gitlab.bj.sensetime.com/open-XLab/cell/nerveX.git
     cd nerveX


3. Install
************

.. code-block:: bash

     # activate environment in server cluster(optional)
     source r0.3.2

     # install for use(if you only want to use nervex)
     # Note: use `--user` option to install the related packages in the user own directory(e.g.: ~/.local)
     pip install . --user
     
     # install for development(if you want to modify and use nervex)
     pip install -e . --user

.. note:: 
    
    you can also install this project in your own virtual environment

4. Check install
****************

After installation, you can open your python console and run the following codes

.. code-block:: python

    import nervex
    print(nervex.__version__)

If the console print the correct version tag, you have successfully installed nerveX

Besides, we also prepare the CLI tool for nervex user, you can type the following command in your terminal

.. code-block:: bash

   nervex -v

If the terminal return the correct information, you can use this CLI tool for the common train and evaluate, you can type ``nervex -h`` for further usage
