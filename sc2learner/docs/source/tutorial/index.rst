Tutorial
===============================

.. toctree::
   :maxdepth: 2


Training
---------

1. Make sure that you have built POD

2. Create directory for one experiment

.. code-block:: bash

    mkdir -p experiments/R50-FPN

3. Copy config file into your workspace

.. code-block:: bash
    
    cp scripts/train.sh scripts/test.sh configs/baselines/faster-rcnn-R50-FPN-1x.yaml experiments/R50-FPN
    cd experiments/R50-FPN

4. Set **ROOT** path and **cfg** path in **train.sh**. By default, it should look like below

.. code-block:: bash

    #!/bin/bash

    T=`date +%m%d%H%M`
    ROOT=../../ # changed
    cfg=faster-rcnn-R50-FPN-1x.yaml # changed

    export PYTHONPATH=$ROOT:$PYTHONPATH

    g=$(($2<8?$2:8))
    srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:$g --ntasks-per-node=$g \
        --job-name=R50-FPN \
    python $ROOT/tools/train_val.py \
    --config=$cfg \
    2>&1 | tee train.log.$T

5. Start training your model

.. code-block:: bash 

    # ./train.sh <PARTITION> <num_gpu> <cfg_path>
    ./train.sh Test 8 faster-rcnn-R50-FPN-1x.yaml
    

You can use **sinfo** to inspect avaliable partitions.

Testing
---------

1. Configure checkpoint 

.. code-block:: yaml

    saver: # Required.
        save_dir: checkpoints
        resume_model: checkpoints/ckpt_e13.pth # checkpoint to test

2. Set **ROOT** path and **cfg** path in **test.sh**. By default, it should look like below

.. note:: 
    make sure you set `-e` option

.. code-block:: bash

    #!/bin/bash
    
    T=`date +%m%d%H%M`
    ROOT=../../
    cfg=faster-rcnn-R50-FPN-1x.yaml
    
    export PYTHONPATH=$ROOT:$PYTHONPATH
    
    g=$(($2<8?$2:8))
    srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:$g --ntasks-per-node=$g \
        --job-name=R50-FPN \
    python $ROOT/tools/train_val.py \
      -e \
      --config=$cfg \
      2>&1 | tee test.log.$T
    
3. Start testing

.. code-block:: bash
 
    # ./test.sh <PARTITION> <num_gpu> <cfg_path>
    ./test.sh Test 8 faster-rcnn-R50-FPN-1x.yaml


Visualization
----------------

目前POD主要支持两种可视化方式

1. 在训练过程中把loss, accuracy等信息输出到tensorboard

.. code-block:: bash 
 
   source r0.2.1
   cd $EXP_DIR # 实验目录文件夹
   tensorboard --logdir=log --port=23333

2. 在训练或者测试时把groud truth或者检测结果显示到图像上

在config文件中的 :ref:`SaverAnchor` 部分配置vis_dt或者vis_gt

.. note::

    对检测结果进行可视化的时候需要指定resume_model和vis_dt

    对ground truth进行可视化的时候只需要指定vis_gt即可

.. code-block:: yaml

    saver: # Required.
        save_dir: checkpoints # dir to save checkpoints
        #pretrain_model: /mnt/lustre/share/DSK/model_zoo/pytorch/imagenet/resnet50-19c8e357.pth
        resume_model: checkpoints/ckpt_e13.pth
        results_dir: results_dir  # dir to save detection results. i.e., bboxes, masks, keypoints
        logdir: log # 用于存储和tensorboard相关的event文件
        #vis_gt:
        #  output_dir: vis_gt
        vis_dt:
            output_dir: vis_dt
            bbox_thresh: 0.3
            keyp_thresh: 0.01
            mask_thresh: 0.5


ToCaffe
---------

POD支持将检测模型转换成caffemodel

**Prerequisite**

安装nart_tools >= v1.0.1-rc6:

**Steps**

1. dataset.class_names: dataset需要提供class_names字段
2. saver.resume_model: 在saver的resume_model指定检测模型路径
3. xxx_head.cfg.tocaffe: 在需要转换的head模块增加tocaffe: True
4. 运行转换脚本，参考scripts/tocaffe.sh
5. 得到xxxx.caffemodel，xxxx.prototxt，当前目录下anchors.json
6. merge BN 等，参考nart_tools
7. 交付，提交prototxt+caffemodel+yaml+anchors.json即可。


Configuration File Generation
------------------------------

为了方便生成常见的配置，POD提供了一个生成配置模板的脚本。

**功能**

该脚本可以用于生成各种模型和方法的配置组合，用户可以基于生成的配置模板进一步细化配置细节。

**用法**

shell脚本见{PROJECT_DIR}/scrips/template.sh，代码文件位于{PROJECT_DIR}/tools/easy_cfg.py。通过一下方式查看用法。


.. code-block:: bash

    PYTHONPATH=.:$PYTHONPATH python tools/easy_cfg.py -h
    usage: easy_cfg.py [-h] [--dataset {COCO,VOC,Custom}]
                       [--num_classes NUM_CLASSES]
                       [--backbone {resnet18,resnet34,resnet50,resnet101,resnet152,resnext_101_32x4d,resnext_101_32x8d,resnext_101_64x4d,resnext_101_64x8d,resnext_152_32x4d,resnext_152_32x8d,resnext_152_64x4d,resnext_152_64x8d,senet154,se_resnet50,se_resnet101,se_res
    net152,se_resnext50_32x4d,se_resnext101_32x4d,se_resnext101_64x4d,nasnetAlarge6_3072,bqnnv1_large,mobilenetv2,shufflenetv2}]
                       [--feature {FPN,C4,C5}] [--first_stage {RPN,RetinaNet}]
                       [--second_stage {None,FC,Res5,RFCN,Cascade}]
                       [--multi_task_stage {None,Mask,Keypoint,Grid}]
                       [--IterMultiplier ITERMULTIPLIER] [--OHEM] [--DCNv2] [--L1]
                       [--FP16] [--SoftNMS] [--MSTest] [--SyncBN] [--flow_style]
     
    implementation of PyTorch Object Detection
     
    optional arguments:
      -h, --help            show this help message and exit
      --dataset {COCO,VOC,Custom}
                            dataset type
      --num_classes NUM_CLASSES
                            number of classes including background, required when
                            dataset is Custom
      --backbone {resnet18,resnet34,resnet50,resnet101,resnet152,resnext_101_32x4d,resnext_101_32x8d,resnext_101_64x4d,resnext_101_64x8d,resnext_152_32x4d,resnext_152_32x8d,resnext_152_64x4d,resnext_152_64x8d,senet154,se_resnet50,se_resnet101,se_resnet152,se_resnext5
    0_32x4d,se_resnext101_32x4d,se_resnext101_64x4d,nasnetAlarge6_3072,bqnnv1_large,mobilenetv2,shufflenetv2}
                            backbone for detector
      --feature {FPN,C4,C5}
                            features for detector
      --first_stage {RPN,RetinaNet}
                            head of the first stage
      --second_stage {None,FC,Res5,RFCN,Cascade}
                            head of the second stage
      --multi_task_stage {None,Mask,Keypoint,Grid}
                            head of multi-task. Grid RCNN is a two stage
                            detector,but it is implemented as a cascade multi-task
                            head
      --IterMultiplier ITERMULTIPLIER
                            Multipliter of iterations
      --OHEM                Use OHEM for training
      --DCNv2               Use DCNv2 for training
      --L1                  Use L1 loss for localization training
      --FP16                Use FP16 for training
      --SoftNMS             Use SoftNMS for testing
      --MSTest              Use multi-scale test
      --SyncBN              SyncBN in detector; if multi-task is Grid, only SyncBN
                            in Grid head
      --flow_style          yaml dump style
    
    
Optional: Testing without evaluation
-------------------------------------------

1. config dataset type as **image_folder**
2. set **image_dir** and supported **exts**

.. code-block:: 

    dataset: # Required.
        type: image_folder
        test:
            image_dir: /mnt/lustre/share/DSK/datasets/mscoco2017/val2017
            flip: False
        exts: ['jpg', 'jpeg', 'png'] # extensions used to filter out non-image files
        alignment: 32              # Align size of images to fit FPN. For fpn-faster-rcnn, 32 is enough; for RetinaNet, it's 128.
        scales: [800]              # shorter side of resized image
        max_size: 1333             # longer side of resized image
        pixel_mean: [0.485, 0.456, 0.406] # ImageNet pretrained statics
        pixel_std: [0.229, 0.224, 0.225]
        batch_size: 2
        workers: 4                 # number of workers of dataloader for each process


.. note::

    When testing images without evalution, you would like to save the detection results. You can 
    config **vis_dt** in **Saver** to enable visualization.
















