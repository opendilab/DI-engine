Configuration
==============

.. toctree::
   :maxdepth: 2


Datasets
--------------

目前主要支持以下数据类型

* :ref:`COCOAnchor`
* :ref:`PASCALAnchor`
* :ref:`CustomDatasetAnchor`

.. note::

    * 为了方便适应产品化对不同评价指标的需求，目前对数据集和评价指标进行了分离。 通过在dataset.evaluator字段下配置选择不同的评价指标。
    * 目前支持的评价指标有COCO (for COCO only)， VOC (for VOC only)， mAP（for CustomDataset only）

.. _COCOAnchor:

**COCO**
~~~~~~~~

.. code-block:: yaml

    dataset: # Required.
      type: coco
      train:
          meta_file: /mnt/lustre/share/DSK/datasets/mscoco2017/annotations/instances_train2017.json
          image_dir: /mnt/lustre/share/DSK/datasets/mscoco2017/train2017
          flip: True
          scales: [800]              # shorter side of resized image
          max_size: 1333             # longer side of resized image
      test:
          meta_file: /mnt/lustre/share/DSK/datasets/mscoco2017/annotations/instances_val2017.json
          image_dir: /mnt/lustre/share/DSK/datasets/mscoco2017/val2017
          flip: False
          scales: [800]              # shorter side of resized image
          max_size: 1333             # longer side of resized image
      num_classes: 81
      has_keypoint: False        # don't use keypoints
      has_mask: False            # don't use segmentations
      aspect_grouping: [1, ]     # use group_sampler when loading dataset
      alignment: 32              # Align size of images to fit FPN. For fpn-faster-rcnn, 32 is enough; for RetinaNet, it's 128.
      pixel_mean: [0.485, 0.456, 0.406] # ImageNet pretrained statics
      pixel_std: [0.229, 0.224, 0.225]
      batch_size: 2
      workers: 4                 # number of workers of dataloader for each process
      evaluator:
          type: COCO               # choices = {'COCO', 'VOC', 'mAP'} 
          kwargs:
              gt_file: /mnt/lustre/share/DSK/datasets/mscoco2017/annotations/instances_val2017.json
              iou_types: [bbox]      # calculate AP_bbox only
    
.. _PASCALAnchor:

**PASCAL VOC**
~~~~~~~~~~~~~~~

.. code-block:: yaml

    dataset: # Required.
      type: pascal_voc
      class_names: [
              "__background__",
              "aeroplane", "bicycle", "bird", "boat",
              "bottle", "bus", "car", "cat", "chair",
              "cow", "diningtable", "dog", "horse",
              "motorbike", "person", "pottedplant",
              "sheep", "sofa", "train", "tvmonitor"]
      train:
          meta_file: /mnt/lustre/share/DSK/datasets/VOC07+12/ImageSets/Main/trainval.txt
          image_dir: /mnt/lustre/share/DSK/datasets/VOC07+12
          flip: True
      test:
          meta_file: /mnt/lustre/share/DSK/datasets/VOC07+12/ImageSets/Main/test.txt
          image_dir: /mnt/lustre/share/DSK/datasets/VOC07+12
          flip: False
      aspect_grouping: [1, ]
      alignment: 1        # Align size of images to fit FPN. For fpn-faster-rcnn, 32 is enough; for RetinaNet, it's 128.
      scales: [600]       # shorter side of resized image
      max_size: 1000      # longer side of resized image
      pixel_mean: [0.485, 0.456, 0.406] # ImageNet pretrained statics
      pixel_std: [0.229, 0.224, 0.225]
      batch_size: 2 
      workers: 4          # number of workers of dataloader for each process
      evaluator:
          type: VOC         # VOC-style metric 
          kwargs:
              gt_file: /mnt/lustre/share/DSK/datasets/VOC07+12/ImageSets/Main/test.txt  # testset image ids
              iou_thresh: 0.5 # mAP0.5


.. _CustomDatasetAnchor:

**Custom Dataset**
~~~~~~~~~~~~~~~~~~~

.. note::

    * 数据标注需要使用专有格式的JSON文件
    * 配置中dataset部分需要添加num_classes字段以进行evaluate
    * 包含class_names字段时可视化显示类别名
    * 将memcached字段设为True将使用memcached，默认为False
    * 设置aspect_grouping来决定图片长宽比的分界点

.. warning::

    使用aspect_grouping对于训练集的shuffle有一定的影响，可能会影响训练效果，目前测试在COCO上不影响性能

.. code-block:: yaml

    dataset: # Required.
      type: custom
      train:
          meta_file: /mnt/lustre/share/DSK/datasets/VOC07+12/example_list/trainval_07+12.json
          image_dir: /mnt/lustre/share/DSK/datasets/VOC07+12/JPEGImages
          flip: True
      test:
          meta_file: /mnt/lustre/share/DSK/datasets/VOC07+12/example_list/test_07.json
          image_dir: /mnt/lustre/share/DSK/datasets/VOC07+12/JPEGImages
          flip: False
      aspect_grouping: [1, ]    # use group_sampler when loading dataset， comment to disable group sampler;
      num_classes: 21
      class_names: [
              "__background__",
              "aeroplane", "bicycle", "bird", "boat",
              "bottle", "bus", "car", "cat", "chair",
              "cow", "diningtable", "dog", "horse",
              "motorbike", "person", "pottedplant",
              "sheep", "sofa", "train", "tvmonitor"]
      alignment: 1              # Align size of images to fit FPN. e.g. H = ceil(H / alignment) * alignment
      scales: [600]             # shorter side of resized image
      max_size: 1000            # longer side of resized image
      pixel_mean: [0.485, 0.456, 0.406]   # ImageNet pretrained statics
      pixel_std: [0.229, 0.224, 0.225]
      batch_size: 2             # batch_size per gpu
      workers: 4                # number of workers of dataloader for each process
      evaluator:
          type: mAP               # calculate mAP metric
          kwargs:
              gt_file: /mnt/lustre/share/DSK/datasets/VOC07+12/example_list/test_07.json # annotation file
              num_classes: 21       # 21 classes including background class
              iou_thresh: 0.5       # mAP0.5


**Annotation json file format for custom dataset**

.. warning:: 

    提供的json文件必须每一行为一条单独的记录，单条记录不能展开为多行。

.. code-block:: javascript

    {"filename": "000005.jpg", "image_height": 375, "image_width": 500, "instances": [{"is_ignored": false, "bbox": [262.0, 210.0, 323.0, 338.0], "label": 9}, {"is_ignored": false, "bbox": [164.0, 263.0, 252.0, 371.0], "label": 9}, {"is_ignored": false, "bbox":[4.0, 243.0, 66.0, 373.0], "label": 9}, {"is_ignored": false, "bbox": [240.0, 193.0, 294.0, 298.0], "label": 9}, {"is_ignored": false, "bbox": [276.0, 185.0, 311.0, 219.0], "label": 9}]}

下面是每条记录便于查看的展开形式

.. code-block:: javascript

   {
      "filename": "000005.jpg",     // Required, 图片路径
      "image_height": 375,          // Optional，图片高度，提供图片高和宽可以使用group sampler对训练加速
      "image_width": 500,           // Optional，图片宽度
      "instances": [                // 标注的实体列表，对于测试集可为空
        {
          "is_ignored": false,       // 是否为忽略区域
          "bbox": [262,210,323,338], // x1,y1,x2,y2
          "label": 9                 // 若为C个类，则label应属于{1,2,...,C}, is_ignored为True时可忽略
        },
        { 
          "is_ignored": false,
          "bbox": [164,263,252,371],
          "label": 9
        },
        {
          "is_ignored": false,
          "bbox": [4,243,66,373],
          "label": 9
        }
      ]
   }

Trainer
------------

该部分用于控制训练过程，包括warmup策略，优化算法，学习率调整等

.. note::

  * 如果使用 warmup, warmup初始学习率等于lr, 
  * warmup结束后的学习率为lr * total_batch_size, total_batch_size等于datasets中的batch size * gpu的数量, 默认8张卡(每张batch size为2)的情况下，学习率等于0.00125*16=0.02


.. note:: 

    * 如果不使用warmup, 初始学习率不会以total_batch_size进行调整

.. code-block:: yaml

    trainer: # Required.
        max_epoch: 13 # total epochs for the training
        test_freq: 14 # test every 14 epochs (当大于max_epoch，则只在训练结束时进行测试）
        optimizer: 
            type: SGD
            kwargs:
                lr: 0.00125
                momentum: 0.9
                weight_decay: 0.0001
        lr_scheduler: # lr_scheduler = MultStepLR(optimizer, milestones=[9,14],gamma=0.1)
            warmup_epochs: 1 # set to be 0 to disable warmup. 
            type: MultiStepLR
            kwargs:
                milestones: [8,11] # epochs to decay lr
                gamma: 0.1 # decay rate

.. _SaverAnchor:

Saver
-----------

存储模块

.. warning::
   
   * vis_gt: 是否在训练过程把gt box的可视化结果存下
   * vis_dt: 是否在测试过程把detection result的可视化结果存下
   * vis_gt 和 vis_dt主要用于调试，一般情况注释掉


.. code-block:: yaml

    saver: # Required.
        save_dir: checkpoints # dir to save checkpoints
        pretrain_model: /mnt/lustre/share/DSK/model_zoo/pytorch/imagenet/resnet50-19c8e357.pth
        #resume_model: checkpoints/ckpt_e13.pth
        results_dir: results_dir  # dir to save detection results. i.e., bboxes, masks, keypoints
        logdir: log # 用于存储和tensorboard相关的event文件
        #vis_gt:
        #  output_dir: vis_gt
        #vis_dt:
        #  output_dir: vis_dt
        #  bbox_thresh: 0.3
        #  keyp_thresh: 0.01
        #  mask_thresh: 0.5

训练过程中POD会用tensorboard维护训练日志，日志包括loss，accuracy和evaluate的指标等。
日志文件目录名由saver.logdir字段指定，默认为“log”

打开tensorboard

.. code-block:: bash 
 
   source r0.2.1
   cd $EXP_DIR # 实验目录文件夹
   tensorboard --logdir=log --port=23333


BatchNorm
----------

.. note::

  bn不再支持全局设置，现在可以对每个模块单独进行bn配置。支持bn的模块有

  * backbones(resnet, resnext, mobilenet, shfflenet...)
  * grid_head(ConvUp)
  * bbox_head(Res5)

batchnorm支持三种模式

* freeze： 固定mean和var
* solo：单卡统计mean和var，不同步
* sync：多卡同步mean和var

.. code-block:: yaml

    bn:
      freeze: true
    # or
    bn:
      solo: true
    # or
    bn:
      sync:
        bn_group_size: 8
        #bn_momentum: 0.1 # r0.2.1环境中使用默认momentum即可

GradClipper
-------------

用于梯度裁剪，支持两种模式。

自动clip并不适用所有模型，部分模型初始阶段的梯度并不具有代表性。建议手动设置clip。

.. code-block:: yaml

    # set max_norm manually
    grad_clipper:
      max_norm: 10
      norm_type: 2
    # or
    # compute max_norm automatically by watching first 10 iterations
    grad_clipper:
      norm_type: 2
      watch_iter: 10

FP16
-------

精度，速度和显存

* 精度：使用fp16在mask-rcnn，keypoint-rcnn，retinanet，faster-rcnn等可以取得fp32同等精度。
* 速度：fp16目前在检测任务中没有明显的加速效果。
* 显存：目前节省显存约20%~30%。训练过程中占用显存越大，fp16节省显存越明显。

.. note::

    在V100上使用FP16可以提高30%左右的训练速度

原理 & 实现

* fp16是指使用16位浮点数于参数的训练和保存，fp32是指的使用32位浮点数于参数的训练和保存，V100对fp16运算有特别的优化，所以使用fp16可以达到训练加速的效果。直接使用fp16会导致部分梯度置0，导致精度损失。实际过程中，使用fp32拷贝来保存模型，使用scale_factor来改变fp16的数值范围。

* forward的过程，bn层、loss计算是使用fp32进行的，其余使用fp16进行计算；backward的过程，将fp16参数的grad拷贝到fp32参数的拷贝上，optimizer.step更新fp32参数，最后把fp32参数拷贝回fp16参数上。

.. warning:: 

    * 模型中对数值精度要求较高的计算不可使用fp16
    * 对SENet使用fp16训练会不收敛，可以考虑将SENet内部的sigmoid使用fp32计算，卷积等使用fp16进行解决TODO

.. code-block:: yaml

    fp16:
     scale_factor: 1024 # factor of shifting the value of graddients into fp16 range

Multi-scale Test 
-----------------

* Multi-Scale Test由MSEnsemble类实现，将detector当作黑盒，对各个模型(Faster-RCNN, Grid-RCNN, Cascade-RCNN)均适用，不需要单独适配
* 目前只适用bbox，暂时不支持Mask-RCNN和Keypoint-RCNN

.. note:: 
    * Multi-scale Test 只支持batch size=1
    * 数据集不需要指定scale，设置为-1
    * 对于单个scale，不使用nms，需要把多个scale下的box放在一起nms

.. code-block::
    
    dataset:
      ...
      test:
        ...
        batch_size: 1  # Multi-scale Test only support batch=1
        scales: [-1]   # Do not resize and flip images in dataset, MSEnsemble need the original image
        flip: False    # to do the augmentation.
    ...
    net:
      ...
      - name: grid_head  # or bbox_head or cascade head
        prev: neck
        type: pod.models.heads.grid_head.ConvUp
        kwargs:
          ...
          cfg:
            ...
            test:       # in MS mode, we filter bboxes with low confidence without NMS.
              nms:
                type: naive
                nms_iou_thresh: -1
              bbox_score_thresh: 0.05
            top_n: -1
     
     
    ms_test:                   # Multi-scale Test config
      bbox_aug:                # bbox test augment, alonewith softnms and box voting
        scales: [400, 500, 600, 700, 800, 900, 1000, 1100, 1200] # multi scales
        max_size: 2000
        hflip: True            # for each scale, do horizontal flipping.
        alignment: 32          # keep alignment consistent with dataset.alignment
      #bbox_vote:
      #  vote_th: 0.9            # [0.0, 1.0]
      #  scoring_method: id      # id, temp_avg, avg, iou_avg, generalized_avg, quasi_sum
      nms:                       # Merge bboxes from all scales
        type: naive
        nms_iou_thresh: 0.5
      bbox_score_thresh: 0.05    # filter out results with low confidence
      top_n: 100                 # number of bboxes to keep


    








