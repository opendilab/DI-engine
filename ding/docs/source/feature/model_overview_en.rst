Model Overview
===================

Neural network, which is often called ``model`` in DI-engine, is one of the most important modules in the whole pipelines. It is usually a part of RL policy or reward model. Generally, DI-engine recommands to pack up all 
the trainable neural network parameters in ``model`` and other parameter-free operations like calculating loss and selecting action are placed in ``policy`` .

Due to the complexity and diversity of neural network 
design and obs/action space, it is hard to abstract a series of general models for each algorithms. Therefore, we design some basic modules in ``ding/model/common`` and ``ding/torch_utils/network`` , and offer 
serveral template models for different algorithms in ``ding/model/template``. Users can set the arguments of template models in some simple environments and implement their own networks referring to the interfaces of template models. Here is a structure graph about the mentioned scheme:

.. image::
   images/model_overview.svg
   :align: center

.. note::
    For the input/output arguments of ``forward`` and related methods, DI-engine first provides defailed information(such as object type and tensor shape) in code comment, and we agree on the following rules:
    
    1. For input, if the input is a torch.Tensor variable, DI-engine directly pass it into model directly, and for other compound inputs, DI-engine uses dict-type input and must record the specific keys in comment.
    2. For output, it is always dict-type output variable, and also need to reocrd the related keys.

.. tip::
    There are often multiple neural networks in a RL algorithms. DI-engine recommands the following design rules:

    1. For the same/similar network architecture but different usage, e.g.: target network in Double DQN and teacher network in Knowledge Distillation, we recommands to use the related model_wrapper.
    2. For the different architectures, such as actor-critic, we recommands that different networks should be different member variables in the whole model.

.. note::
    As to code implementation and arrangement, DI-engine use the following rules:

    1. If a neural network is utilized in more than 2 algorithms and environments, it shoule be added into the main repo. Otherwise, the specific use site is a better choice.
    2. If a neural network is a kind of general deep learning module, it should be located in ``ding/torch_utils/network``.
    3. For some RL specific networks, such as DuelingHead, it shoule be implemented in ``ding/model/commom``.
