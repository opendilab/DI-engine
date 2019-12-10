Model Flow
===========

.. toctree::
   :maxdepth: 2


模型由若干子模块构成，子模块内部实现互相独立，子模块之间的依赖由约定的接口实现。

模块之间的输入输出均为字典类型

Detection Model
~~~~~~~~~~~~~~~~

检测模型被抽象为“特征提取器 + 任务分支”的结构。特征提取器包括 backbone 和 neck，任务分支为不同的heads

**BackBone & Neck**

任何的Backbone或者Neck需要继承于 :class:`torch.nn.Module` , 需要实现以下几个接口:

* :meth:`~pod.models.backbones.ResNet.__init__` 当模块有前驱时，第一个参数为前趋模块输出channel数
* :meth:`~pod.models.backbones.ResNet.get_outplanes` 当模块有后继时，需实现此方法返回该模块的输出channel数，帮助构建后继网络
* :meth:`~pod.models.backbones.ResNet.forward` input为dataset的输出，该方法的输出格式一般为

.. code-block:: python 

    {'features':[], 'strides': []}

**Head**

Head模块需要继承-:class:`torch.nn.Module`，主要是处理经过Backbone和Neck之后的数据

需要实现以下几个接口:

* :meth:`~pod.models.heads.bbox_head.bbox_head.BboxNet.__init__` 当模块有前驱时，第一个参数为前趋模块输出channel数
* :meth:`~pod.models.heads.bbox_head.bbox_head.BboxNet.forward` input为backbone或者neck的输出，该方法的输出一般为

.. code-block:: python

   {
     # ... 前面所有模块的输出
     'dt_bboxes': [], # 检测框, RoINet和BboxNet的输出
     'dt_keyps': [], # 检测框对应的keypoints，KeypNet的输出
     'dt_masks': [] # 检测框对应的segmentation，MaskNet的输出
   }

.. note:: 

    采用了算法和网络结构分离的设计, 基类(RoINet, BboxNet, KeypNet, MaskNet)实现算法, 子类(NaiveRPN, FC, RFCN, ConvUp)实现具体网络结构

Dataset
~~~~~~~~~~~~~~~

所有类型的Dataset都要继承于 :class:`~pod.datasets.base_dataset.BaseDataset` 需要实现以下几个接口

* :meth:`~pod.datasets.base_dataset.BaseDataset.__len__` 返回Dataset的长度
* :meth:`~pod.datasets.base_dataset.BaseDataset.__getitem__` 返回单条数据记录, 使用字典，数据内容与之后的backbone的读入保证一致即可
* :meth:`~pod.datasets.base_dataset.BaseDataset.evaluate` 计算评价指标
* :meth:`~pod.datasets.base_dataset.BaseDataset.dump` 把预测结果写入文件
* :meth:`~pod.datasets.base_dataset.BaseDataset.vis_gt` 可视化ground truth
* :meth:`~pod.datasets.base_dataset.BaseDataset.vis_dt` 可视化检测结果



















