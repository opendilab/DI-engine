DI-orchestrator Overview
==========================


In order to provide running support for DI-engine in Kubernetes (K8s), we designed DI-orchestrator, which aims to manager all the modules in the distributed training of DI-engine. DI-orchestrator offers many micro 
services for more stable and efficient training. The detailed architecture design is shown in `DI-orchestrator Guide <https://github.com/opendilab/DI-orchestrator/blob/master/docs/architecture.md>`_. And here is a 
specific example about how to launch a DI-engine training job(``DIJob``), and first you should deploy a k8s cluster first:

.. _1-submit-view-modify-and-delete-dijob:

1. Submit, View, Modify and Delete DIJob
----------------------------------------

A simple example is stored in
``{DING_ROOT}/ding/scripts/dijob-qbert.yaml``, you can use this example
to learn how to submit a ``DIJob`` on Kubenetes cluster.

.. code:: shell

   # submit DIJob
   kubectl create -f dijob-qbert.yaml
   diengine.opendilab.org/qbert-dqn created

   # get pod and you will see coordinator are created
   NAME                              READY   STATUS    RESTARTS   AGE 
   qbert-dqn-coordinator   					1/1     Running   0          8s

   # few seconds later, you will see collectors and learners (and aggregator if need) created by di-server
   $ kubectl get pod
   NAME                                  READY   STATUS    RESTARTS   AGE
   qbert-dqn-aggregator        					1/1     Running   0          80s
   qbert-dqn-collector-pm5gv   					1/1     Running   0          66s
   qbert-dqn-coordinator       					1/1     Running   0          80s
   qbert-dqn-learner-rcwmc     					1/1     Running   0          66s
   qbert-dqn-learner-txjks     					1/1     Running   0          66s

   # get logs
   $ kubectl logs qbert-dqn-coordinator
   * Serving Flask app "interaction.master.master" (lazy loading)
    * Environment: production
      WARNING: This is a development server. Do not use it in a production deployment.
      Use a production WSGI server instead.
    * Debug mode: off
   ...

   # delete DIJob
   $ kubectl delete dijob qbert-dqn
   # or
   $ kubectl delete -f qbert-dqn.yaml

.. _2-check-the-status-of-dijob:

2. Check the status of DIJob
----------------------------

.. code:: shell

   # get the dijob qbert-dqn to display
   $ kubectl get dijob dijob-example
   NAME            PHASE       AGE
   qbert-dqn   Succeeded   22h

   # show details of a specific resource or group of resource
   $ kubectl describe dijob qbert-dqn
   Name:         qbert-dqn
   Namespace:    default
   Labels:       <none>
   ...
     Phase:                   Succeeded
     Replica Status:
       Aggregator:
       Collector:
         Succeeded:  2
       Coordinator:
         Succeeded:  1
       Learner:
         Succeeded:  1
   Events:           <none> 

.. _3-set-storage-middleware:

3. Set storage middleware
-------------------------

As ``DIJob`` needs storage middleware during training, it must be
provided in ``.yaml``. Here's an example that shows the required fields
for setting a storage middleware.

hostPath configuration example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Supply a directory from the host node's filesystem in the field
   ``volumes`` in the outermost ``spec``. The field ``name`` (here is
   ``work-dir``) and the field ``path`` (directory location on host)
   must be provided are required.

.. code:: yaml

   spec:
     ...
     volumes:
     - name: work-dir
       hostPath:
         path: /data/nfs/ding/qbert

1. In the field ``volumeMounts`` field of worker, fill the field
   ``name`` and ``mountPath`` to specify that the destination inside the
   pod a volume gets mounted to. Note that the name of ``volumeMounts``
   must be the same as the name defined in ``hostPath``.

.. code:: yaml

     ...
     coordinator:
       template:
         spec:
           containers:
           ...
             volumeMounts:
             - name: work-dir
               mountPath: /ding
   ...

.. _4-insert-experimental-config-in-dijob-config:

4. Insert experimental config in DIJob config
---------------------------------------------

Generally, the experimental config (e.g. ``cartpole_dqn_config.py``) and
``DIJob`` config are stored in two different files. You can use the
following two methods to launch your experiment:

1. Place the experimental config file in the mounted volume in advance,
   as in the above-mentioned ``/data/nfs/ding/qbert``;

2. Insert experimental config in the ``DIJob`` config;

Here's an example of Inserting experimental config in the ``DIJob``
config:

.. code:: yaml

     coordinator:
       template:
         spec:
           containers:
           - name: coordinator
             image: ...
             ...
             command: ["/bin/bash", "-c",]
             args:
             - |
               cat <<EOF > qbert_dqn_config_k8s.py
               from easydict import EasyDict

               qbert_dqn_config = dict(
                   env=dict(
                       collector_env_num=16,
                       collector_episode_num=2,
                       evaluator_env_num=8,
                       evaluator_episode_num=1,
                       stop_value=30000,
                       env_id='QbertNoFrameskip-v4',
                       frame_stack=4,
                       manager=dict(
                           shared_memory=False,
               ...
               ...
               qbert_dqn_system_config = EasyDict(qbert_dqn_system_config)
               system_config = qbert_dqn_system_config
               EOF

               ding -m dist --module config -p k8s -c qbert_dqn_config_k8s.py -s 0;
               ding -m dist --module coordinator -p k8s -c qbert_dqn_config_k8s.py.pkl -s 0
           ...

.. _5-define-environment-variables-for-a-worker:

5. Define environment variables for a worker
--------------------------------------------

To set environment variables, include the ``env`` field in the
configuration file. Here's an example of defining an environment
variable with name ``PYTHONUNBUFFERED`` and value ``1``\ ：

.. code:: yaml

     ...
     coordinator:
       template:
         spec:
           containers:
           - name: coordinator
             image: ..
             ...
             env:
             - name: PYTHONUNBUFFERED
               value: "1"
             ...
      ...

.. _6-assign-cpu-memory-and-gpu-resources-to-workers:

6. Assign CPU, memory, and GPU resources to workers
---------------------------------------------------

The CPU, memory, and GPU required by each worker may be different. You
need to specify requests in the field ``resources:requests`` of each
worker. To specify a resource limit, include ``resources:limits``.

Here's an example of the configuration file for the learner which has a
request of 6 CPU, 1 GPU, 10Gi memory and a limit of 6 CPU, 1 GPU, 10Gi
memory:

.. code:: yaml

   ...
     learner:
       template:
         spec:
           containers:
           - name: learner
             image: ...
             ...
             resources:
               requests:
                 cpu: "6"
                 nvidia.com/gpu: "1"
                 memory: "10Gi"
               limits:
                 cpu: "6"
                 nvidia.com/gpu: "1"
                 memory: "10Gi"
