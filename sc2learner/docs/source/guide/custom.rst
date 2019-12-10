Custom Module
==============

.. toctree::
   :maxdepth: 2


Backbone
----------

实现一个最简单的Backbone只需要完成三个函数, :meth:`__init__`, :meth:`get_outplanes`, :meth:`forward`
其中 :meth:`__init__` 函数的参数主要取决于配置文件, 例如，我们先使用一个最简单的配置文件 `custom_net.yaml`, 实现一个L层的卷积网络

.. note::

    确保config文件中定义的type为能够被import的模块，在这个例子中，我们新建一个文件custom.py放在 `pod.models.backbones` 文件夹下面

.. code-block:: yaml

   net: 
        name: backbone
        type: pod.models.backbones.custom.CustomNet
        kwarg:
            depth: 3   
            out_planes: [64, 128, 256]


.. code-block:: python

    import torch
    import torch.nn as nn

    class CustomNet(torch.nn.Module):
        def __init__(self, depth, out_planes):
        """
        构造参数为配置文件中的kwarg, 在这个例子中由于没有前驱模块，
        所以没有inplances参数
        """

        self.out_planes = out_planes[-1]

        in_planes = 3
        for i in range(depth):
            self.add_module(f'layer{i}', 
                            nn.Conv2d(in_planes, out_planes[i], kernel_size=3, padding=1))
            self.add_module('relu', nn.ReLU(inplace=True))
            in_planes = out_planes[i]

然后我们再实现 :meth:`forward` 和 :meth:`get_outplanes` 函数

.. note:: 

    :meth:`foward` 函数需要计算输出的features和strides, 这两个值都为数组形式。

.. code-block:: python

    def forward(self, input):
        """
        input的字典类型，数据的组织方式主要取决于config中定义好的Dataset, 
        在这里我们假设input中包含了image这一项
        """

        x = input['image'] 

        for submodule in self.children():
            x = submodule(x)
        
        # 输出为一个字典，需要包括features和strides两项, 同时我们保留input中的其他数据
        input['features'] = [x]
        input['strides'] = [1]

        return input

    def get_outplanes(self):

        return self.out_planes


Head
-----

实现Head函数只需要实现 :meth:`__init__`, :meth:`forward`. 其中初始化方式和Backbone的初始化一致，取决于config参数。
在这个例子中，我们利用前面定义好的CustomNet, 实现一个CustomHead, 完成一个简单的分类网络


新建文件 **custom.py** 放在 `pod.models.heads` 目录下

config 文件示例

.. code-block:: yaml

   net: 
     - name: backbone
         type: pod.models.backbones.custom.CustomNet
         kwarg:
           depth: 3   
           out_planes: [64, 128, 256]

     - name: head
         prev: backbone
         type: pod.models.heads.custom.CustomHead
         kwarg:
            num_classes: 10
            

我们使用前面自定义的 :class:`CustomNet` 作为前驱, 设置Head的prev

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class CustomHead(nn.Module):
        def __init__(self, in_planes, num_classes):
            """
            由于在配置文件中，我们配置了head有prev部分，因此在构造函数部分会传入in_planes参数
            """

            # build your model.. 

            self.fc = nn.Linear(inplanes, num_classes)

        def forward(self, input):
            """
            input为字典类型，包含了backbone的输出和dataset的输出
            """

            # implement your algorithm
            # 这里简单使用 global average pooling 和一层 FC

            output = input['features'][0].mean(-1).mean(-1)
            output = self.fc(output)

            loss = self._get_loss(output, input['label'])

            # 将loss放入输出字典中，确保能够让POD对其收集并进行backward
            input['ce_loss'] = loss

        def _get_loss(self, out, label):

            return F.cross_entropy(out, label)

.. note::

    POD会在最后的输出的字典中寻找所有的包含有loss的项，对他们进行 :meth:`backward` 操作





