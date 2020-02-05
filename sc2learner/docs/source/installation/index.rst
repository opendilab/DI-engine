Installation
===============================

.. toctree::
   :maxdepth: 3


1. Look up tags and find corresponding historical version (master branch recommend)

.. code-block:: bash

     git clone git@gitlab.bj.sensetime.com:xialiqiao/SenseStar.git
     cd SenseStar


2. Activate environment and install

.. code-block:: bash

     # actiavte environment in lustre
     source r0.3.0

     # install for development
     pip install -e . --user

.. note:: 
    
    you can also install this project in your own virtual environment

3. Prepare SC2 game setting

   - Download SC2 game packages and maps(refer to https://github.com/Blizzard/s2client-proto)
     - Default game version: 4.10

   - Set up environment variable **SC2PATH**

.. code-block:: bash

    # for example you can run the following shell command,
    # or you can add it into your .bashrc or .zshrc
    export SC2PATH=/mnt/lustre/niuyazhe/StarCraftII
