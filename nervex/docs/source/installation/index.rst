Installation
===============================

.. toctree::
   :maxdepth: 3


1. Look up tags and find corresponding historical version (master branch recommend)

.. code-block:: bash

     git clone http://gitlab.bj.sensetime.com/open-XLab/cell/nerveX.git
     cd nerveX


2. Activate environment and install(in server cluster)

.. code-block:: bash

     # actiavte environment in server cluster
     source r0.3.2

     # install for development
     # Note: use `--user` option to install the related packages in the user own directory(e.g.: ~/.local)
     pip install -e . --user

.. note:: 
    
    you can also install this project in your own virtual environment
