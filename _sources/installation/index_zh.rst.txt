安装说明
===============================

.. toctree::
   :maxdepth: 3

前置条件
***********************************

   系统版本:

     - Ubuntu 16.04, 18.04, 20.04
     - Centos Linux version 3.10.0-693.el7.x86_64
     - macOS
     - Windows10

   Python 版本: 3.6, 3.7, 3.8 (你可以参考 `Python 安装指南 <https://pytorch.org/get-started/locally/#linux-python>`_ . 请注意安装正确的Python版本.) 

   PyTorch 版本: >=1.3.1, <=1.8.0 (你可以在这个范围内使用任何合适的版本, 参考 `PyTorch 安装 <https://pytorch.org/get-started/locally/>`_)

   .. note::

        如果您需要使用GPU，建议您使用带有CUDA的PyTorch。否则，您只需要安装cpu版本的PyTorch。


发布版本
**********************************************************

您可以从PyPI使用以下命令安装DI-engine：


.. code-block:: bash

     pip install DI-engine

.. tip::

    如果您在下载软件包时遇到超时错误，您可以尝试采用更换其他pip源（比如 https://pypi.douban.com/simple）。
    

如果您更喜欢使用Anaconda或者Miniconda, 建议您采用如下命令:

.. code-block:: bash

    conda install -c opendilab di-engine

此外，您还可以从github中的源代码安装DI-engine（建议使用main分支）

.. code-block:: bash

    git clone https://github.com/opendilab/DI-engine.git
    cd DI-engine
    pip install . --user

.. tip::

   如果在安装中使用 ``--user`` 选项，则某些可执行命令将安装在用户路径（例如 ``~/.local/bin`` 中），并且您应该确保该路径已经添加到环境变量中（例如“$PATH in Linux”）。

如果您想安装DI-engine中某些功能（如具体某些环境，单元测试工具等）所需的额外包，可以执行：

.. code-block:: bash

     pip install DI-engine[common_env]  # install atari-env and box-2d env
     pip install DI-engine[test]  # install unittest(pytest) related package

如果安装之后显示出了如下所示的内容，则表示安装成功，您可以参考下一节内容进行检查。

.. code-block:: bash

    Installing collected packages: DI-engine
      Running setup.py develop for DI-engine
    Successfully installed DI-engine

.. tip::
    有些shell（如zsh）需要在包名前后加引号，例如 pip install 'DI-engine[test]' 


.. note::
    
   整个安装过程通常持续30秒左右（取决于软件包的下载速度），如果有一些失败的软件包，也可以参考 ``setup.py`` 安装特定的软件包手动安装。


开发版本
********************
为了对DI-engine有所贡献，并支持运行测试，您需要找到合适的发布标签或主分支

.. code-block:: bash

    # source r0.3.2  # maybe you need activate virtual environment first

    git clone https://github.com/opendilab/DI-engine.git && cd DI-engine
    pip install -e .[test, common_env] --user

检查安装情况
****************

安装之后，可以打开python控制台并运行以下代码

.. code-block:: python

    import ding
    print(ding.__version__)

如果控制台打印了正确的版本标签，则表示您已成功安装DI-engine

此外，DI-engine还为用户准备了CLI工具，您可以在终端中键入以下命令：

.. code-block:: bash

   ding -v

如果终端返回正确的信息，您可以使用这个CLI工具进行常见的训练和评估，您可以键入 ``ding -h`` 查看更多帮助。
