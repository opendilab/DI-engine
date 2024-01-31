ding.utils
--------------


autolog
========
Please refer to ``ding/utils/autolog`` for more details.

TimeMode
~~~~~~~~
.. autoclass:: ding.utils.autolog.TimeMode
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

RangedData
~~~~~~~~~~~
.. autoclass:: ding.utils.autolog.RangedData
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

TimeRangedData
~~~~~~~~~~~~~~
.. autoclass:: ding.utils.autolog.TimeRangedData
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

LoggedModel
~~~~~~~~~~~
.. autoclass:: ding.utils.autolog.LoggedModel
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

BaseTime
~~~~~~~~~
.. autoclass:: ding.utils.autolog.BaseTime
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

NaturalTime
~~~~~~~~~~~~~
.. autoclass:: ding.utils.autolog.NaturalTime
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

TickTime
~~~~~~~~~
.. autoclass:: ding.utils.autolog.TickTime
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

TimeProxy
~~~~~~~~~
.. autoclass:: ding.utils.autolog.TimeProxy
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

LoggedValue
~~~~~~~~~~~~
.. autoclass:: ding.utils.autolog.LoggedValue
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

data.structure
==============
Please refer to ``ding/utils/data/structure`` for more details.

Cache
~~~~~~~
.. autoclass:: ding.utils.data.structure.Cache
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

LifoDeque
~~~~~~~~~~
.. autoclass:: ding.utils.data.structure.LifoDeque
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

data.base_dataloader
====================
Please refer to ``ding/utils/data/base_dataloader`` for more details.

IDataLoader
~~~~~~~~~~~
.. autoclass:: ding.utils.data.base_dataloader.IDataLoader
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

data.collate_fn
=================
Please refer to ``ding/utils/data/collate_fn`` for more details.

ttorch_collate
~~~~~~~~~~~~~~
.. autofunction:: ding.utils.data.collate_fn.ttorch_collate

default_collate
~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.data.collate_fn.default_collate

timestep_collate
~~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.data.collate_fn.timestep_collate

diff_shape_collate
~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.data.collate_fn.diff_shape_collate

default_decollate
~~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.data.collate_fn.default_decollate

data.dataloader
================
Please refer to ``ding/utils/data/dataloader`` for more details.

AsyncDataLoader
~~~~~~~~~~~~~~~
.. autoclass:: ding.utils.data.dataloader.AsyncDataLoader
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

data.dataset
============
Please refer to ``ding/utils/data/dataset`` for more details.

DatasetStatistics
~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.utils.data.dataset.DatasetStatistics
    :special-members: __init__, __len__, __getitem__
    :members:
    :private-members:
    :undoc-members:

NaiveRLDataset
~~~~~~~~~~~~~~
.. autoclass:: ding.utils.data.dataset.NaiveRLDataset
    :special-members: __init__, __len__, __getitem__
    :members:
    :private-members:
    :undoc-members:

D4RLDataset
~~~~~~~~~~~
.. autoclass:: ding.utils.data.dataset.D4RLDataset
    :special-members: __init__, __len__, __getitem__
    :members:
    :private-members:
    :undoc-members:

HDF5Dataset
~~~~~~~~~~~
.. autoclass:: ding.utils.data.dataset.HDF5Dataset
    :special-members: __init__, __len__, __getitem__
    :members:
    :private-members:
    :undoc-members:

D4RLTrajectoryDataset
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.utils.data.dataset.D4RLTrajectoryDataset
    :special-members: __init__, __len__, __getitem__
    :members:
    :private-members:
    :undoc-members:

D4RLDiffuserDataset
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.utils.data.dataset.D4RLDiffuserDataset
    :special-members: __init__, __len__, __getitem__
    :members:
    :private-members:
    :undoc-members:

FixedReplayBuffer
~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.utils.data.dataset.FixedReplayBuffer
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

PCDataset
~~~~~~~~~
.. autoclass:: ding.utils.data.dataset.PCDataset
    :special-members: __init__, __len__, __getitem__
    :members:
    :private-members:
    :undoc-members:

load_bfs_datasets
~~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.data.dataset.load_bfs_datasets


BCODataset
~~~~~~~~~~
.. autoclass:: ding.utils.data.dataset.BCODataset
    :special-members: __init__, __len__, __getitem__
    :members:
    :private-members:
    :undoc-members:

SequenceDataset
~~~~~~~~~~~~~~~
.. autoclass:: ding.utils.data.dataset.SequenceDataset
    :special-members: __init__, __len__, __getitem__
    :members:
    :private-members:
    :undoc-members:

hdf5_save
~~~~~~~~~
.. autofunction:: ding.utils.data.dataset.hdf5_save

naive_save
~~~~~~~~~~~
.. autofunction:: ding.utils.data.dataset.naive_save

offline_data_save_type
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.data.dataset.offline_data_save_type

create_dataset
~~~~~~~~~~~~~~
.. autofunction:: ding.utils.data.dataset.create_dataset



bfs_helper
==========
Please refer to ``ding/utils/bfs_helper`` for more details.

get_vi_sequence
~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.bfs_helper.get_vi_sequence

collection_helper
=================
Please refer to ``ding/utils/collection_helper`` for more details.

iter_mapping
~~~~~~~~~~~~
.. autofunction:: ding.utils.collection_helper.iter_mapping

compression_helper
==================
Please refer to ``ding/utils/compression_helper`` for more details.

CloudPickleWrapper
~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.utils.compression_helper.CloudPickleWrapper
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

dummy_compressor
~~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.compression_helper.dummy_compressor

zlib_data_compressor
~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.compression_helper.zlib_data_compressor

lz4_data_compressor
~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.compression_helper.lz4_data_compressor

jpeg_data_compressor
~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.compression_helper.jpeg_data_compressor

get_data_compressor
~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.compression_helper.get_data_compressor

dummy_decompressor
~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.compression_helper.dummy_decompressor

lz4_data_decompressor
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.compression_helper.lz4_data_decompressor

zlib_data_decompressor
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.compression_helper.zlib_data_decompressor

jpeg_data_decompressor
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.compression_helper.jpeg_data_decompressor

get_data_decompressor
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.compression_helper.get_data_decompressor

default_helper
==============
Please refer to ``ding/utils/default_helper`` for more details.

get_shape0
~~~~~~~~~~
.. autofunction:: ding.utils.default_helper.get_shape0

lists_to_dicts
~~~~~~~~~~~~~~
.. autofunction:: ding.utils.default_helper.lists_to_dicts

dicts_to_lists
~~~~~~~~~~~~~~
.. autofunction:: ding.utils.default_helper.dicts_to_lists

override
~~~~~~~~
.. autofunction:: ding.utils.default_helper.override

squeeze
~~~~~~~
.. autofunction:: ding.utils.default_helper.squeeze

default_get
~~~~~~~~~~~
.. autofunction:: ding.utils.default_helper.default_get

list_split
~~~~~~~~~~
.. autofunction:: ding.utils.default_helper.list_split

error_wrapper
~~~~~~~~~~~~~
.. autofunction:: ding.utils.default_helper.error_wrapper

LimitedSpaceContainer
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.utils.default_helper.LimitedSpaceContainer
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

deep_merge_dicts
~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.default_helper.deep_merge_dicts

deep_update
~~~~~~~~~~~
.. autofunction:: ding.utils.default_helper.deep_update

flatten_dict
~~~~~~~~~~~~
.. autofunction:: ding.utils.default_helper.flatten_dict

set_pkg_seed
~~~~~~~~~~~~
.. autofunction:: ding.utils.default_helper.set_pkg_seed

one_time_warning
~~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.default_helper.one_time_warning

split_fn
~~~~~~~~
.. autofunction:: ding.utils.default_helper.split_fn

split_data_generator
~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.default_helper.split_data_generator

RunningMeanStd
~~~~~~~~~~~~~~~
.. autoclass:: ding.utils.default_helper.RunningMeanStd
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

make_key_as_identifier
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.default_helper.make_key_as_identifier

remove_illegal_item
~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.default_helper.remove_illegal_item

design_helper
=============
Please refer to ``ding/utils/design_helper`` for more details.

SingletonMetaclass
~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.utils.design_helper.SingletonMetaclass
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

fake_linklink
=============
Please refer to ``ding/utils/fake_linklink`` for more details.

FakeClass
~~~~~~~~~
.. autoclass:: ding.utils.fake_linklink.FakeClass
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

FakeNN
~~~~~~~
.. autoclass:: ding.utils.fake_linklink.FakeNN
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

FakeLink
~~~~~~~~
.. autoclass:: ding.utils.fake_linklink.FakeLink
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

fast_copy
=========
Please refer to ``ding/utils/fast_copy`` for more details.

_FastCopy
~~~~~~~~~
.. autoclass:: ding.utils.fast_copy._FastCopy
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

file_helper
===========
Please refer to ``ding/utils/file_helper`` for more details.

read_from_ceph
~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.file_helper.read_from_ceph

_get_redis
~~~~~~~~~~
.. autofunction:: ding.utils.file_helper._get_redis

read_from_redis
~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.file_helper.read_from_redis

_ensure_rediscluster
~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.file_helper._ensure_rediscluster

read_from_rediscluster
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.file_helper.read_from_rediscluster

read_from_file
~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.file_helper.read_from_file

_ensure_memcached
~~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.file_helper._ensure_memcached

read_from_mc
~~~~~~~~~~~~
.. autofunction:: ding.utils.file_helper.read_from_mc

read_from_path
~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.file_helper.read_from_path

save_file_ceph
~~~~~~~~~~~~~~
.. autofunction:: ding.utils.file_helper.save_file_ceph

save_file_redis
~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.file_helper.save_file_redis

save_file_rediscluster
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.file_helper.save_file_rediscluster

read_file
~~~~~~~~~
.. autofunction:: ding.utils.file_helper.read_file

save_file
~~~~~~~~~
.. autofunction:: ding.utils.file_helper.save_file

remove_file
~~~~~~~~~~~
.. autofunction:: ding.utils.file_helper.remove_file

import_helper
=============
Please refer to ``ding/utils/import_helper`` for more details.

try_import_ceph
~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.import_helper.try_import_ceph

try_import_mc
~~~~~~~~~~~~~
.. autofunction:: ding.utils.import_helper.try_import_mc

try_import_redis
~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.import_helper.try_import_redis

try_import_rediscluster
~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.import_helper.try_import_rediscluster

try_import_link
~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.import_helper.try_import_link

import_module
~~~~~~~~~~~~~~
.. autofunction:: ding.utils.import_helper.import_module

k8s_helper
===========
Please refer to ``ding/utils/k8s_helper`` for more details.

get_operator_server_kwargs
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.k8s_helper.get_operator_server_kwargs

exist_operator_server
~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.k8s_helper.exist_operator_server

pod_exec_command
~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.k8s_helper.pod_exec_command

K8sType
~~~~~~~
.. autoclass:: ding.utils.k8s_helper.K8sType
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

K8sLauncher
~~~~~~~~~~~
.. autoclass:: ding.utils.k8s_helper.K8sLauncher
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:


linklink_dist_helper
====================
Please refer to ``ding/utils/linklink_dist_helper`` for more details.

get_rank
~~~~~~~~
.. autofunction:: ding.utils.linklink_dist_helper.get_rank

get_world_size
~~~~~~~~~~~~~~
.. autofunction:: ding.utils.linklink_dist_helper.get_world_size

broadcast
~~~~~~~~~
.. autofunction:: ding.utils.linklink_dist_helper.broadcast

allreduce
~~~~~~~~~
.. autofunction:: ding.utils.linklink_dist_helper.allreduce

allreduce_async
~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.linklink_dist_helper.allreduce_async

get_group
~~~~~~~~~
.. autofunction:: ding.utils.linklink_dist_helper.get_group

dist_mode
~~~~~~~~~
.. autofunction:: ding.utils.linklink_dist_helper.dist_mode

dist_init
~~~~~~~~~
.. autofunction:: ding.utils.linklink_dist_helper.dist_init

dist_finalize
~~~~~~~~~~~~~
.. autofunction:: ding.utils.linklink_dist_helper.dist_finalize

DistContext
~~~~~~~~~~~~
.. autoclass:: ding.utils.linklink_dist_helper.DistContext
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

simple_group_split
~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.linklink_dist_helper.simple_group_split

synchronize
~~~~~~~~~~~
.. autofunction:: ding.utils.linklink_dist_helper.synchronize

lock_helper
===========
Please refer to ``ding/utils/lock_helper`` for more details.

LockContextType
~~~~~~~~~~~~~~~
.. autoclass:: ding.utils.lock_helper.LockContextType
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

LockContext
~~~~~~~~~~~
.. autoclass:: ding.utils.lock_helper.LockContext
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

get_rw_file_lock
~~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.lock_helper.get_rw_file_lock

FcntlContext
~~~~~~~~~~~~~
.. autoclass:: ding.utils.lock_helper.FcntlContext
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

get_file_lock
~~~~~~~~~~~~~
.. autofunction:: ding.utils.lock_helper.get_file_lock

log_helper
==========
Please refer to ``ding/utils/log_helper`` for more details.

build_logger
~~~~~~~~~~~~~
.. autofunction:: ding.utils.log_helper.build_logger

TBLoggerFactory
~~~~~~~~~~~~~~~
.. autoclass:: ding.utils.log_helper.TBLoggerFactory
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

LoggerFactory
~~~~~~~~~~~~~
.. autoclass:: ding.utils.log_helper.LoggerFactory
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

pretty_print
~~~~~~~~~~~~
.. autofunction:: ding.utils.log_helper.pretty_print


log_writer_helper
=================
Please refer to ``ding/utils/log_writer_helper`` for more details.

DistributedWriter
~~~~~~~~~~~~~~~~~
.. autoclass:: ding.utils.log_writer_helper.DistributedWriter
    :members: plugin, get_instance, __del__

enable_parallel
~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.log_writer_helper.enable_parallel

normalizer_helper
=================
Please refer to ``ding/utils/normalizer_helper`` for more details.

DatasetNormalizer
~~~~~~~~~~~~~~~~~
.. autoclass:: ding.utils.normalizer_helper.DatasetNormalizer
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

flatten
~~~~~~~
.. autofunction:: ding.utils.normalizer_helper.flatten

Normalizer
~~~~~~~~~~~
.. autoclass:: ding.utils.normalizer_helper.Normalizer
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

GaussianNormalizer
~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.utils.normalizer_helper.GaussianNormalizer
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

CDFNormalizer
~~~~~~~~~~~~~~
.. autoclass:: ding.utils.normalizer_helper.CDFNormalizer
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

CDFNormalizer1d
~~~~~~~~~~~~~~~~
.. autoclass:: ding.utils.normalizer_helper.CDFNormalizer1d
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

empirical_cdf
~~~~~~~~~~~~~
.. autofunction:: ding.utils.normalizer_helper.empirical_cdf

atleast_2d
~~~~~~~~~~
.. autofunction:: ding.utils.normalizer_helper.atleast_2d

LimitsNormalizer
~~~~~~~~~~~~~~~~
.. autoclass:: ding.utils.normalizer_helper.LimitsNormalizer
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

orchestrator_launcher
=====================
Please refer to ``ding/utils/orchestrator_launcher`` for more details.

OrchestratorLauncher
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.utils.orchestrator_launcher.OrchestratorLauncher
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

create_components_from_config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.orchestrator_launcher.create_components_from_config

wait_to_be_ready
~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.orchestrator_launcher.wait_to_be_ready

profiler_helper
===============
Please refer to ``ding/utils/profiler_helper`` for more details.

Profiler
~~~~~~~~
.. autoclass:: ding.utils.profiler_helper.Profiler
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

pytorch_ddp_dist_helper
=======================
Please refer to ``ding/utils/pytorch_ddp_dist_helper`` for more details.

get_rank
~~~~~~~~
.. autofunction:: ding.utils.pytorch_ddp_dist_helper.get_rank

get_world_size
~~~~~~~~~~~~~~
.. autofunction:: ding.utils.pytorch_ddp_dist_helper.get_world_size

allreduce
~~~~~~~~~
.. autofunction:: ding.utils.pytorch_ddp_dist_helper.allreduce

allreduce_async
~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.pytorch_ddp_dist_helper.allreduce_async

reduce_data
~~~~~~~~~~~
.. autofunction:: ding.utils.pytorch_ddp_dist_helper.reduce_data

allreduce_data
~~~~~~~~~~~~~~
.. autofunction:: ding.utils.pytorch_ddp_dist_helper.allreduce_data

get_group
~~~~~~~~~
.. autofunction:: ding.utils.pytorch_ddp_dist_helper.get_group

dist_mode
~~~~~~~~~
.. autofunction:: ding.utils.pytorch_ddp_dist_helper.dist_mode

dist_init
~~~~~~~~~
.. autofunction:: ding.utils.pytorch_ddp_dist_helper.dist_init

dist_finalize
~~~~~~~~~~~~~
.. autofunction:: ding.utils.pytorch_ddp_dist_helper.dist_finalize

DDPContext
~~~~~~~~~~
.. autoclass:: ding.utils.pytorch_ddp_dist_helper.DDPContext
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

simple_group_split
~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.pytorch_ddp_dist_helper.simple_group_split

to_ddp_config
~~~~~~~~~~~~~
.. autofunction:: ding.utils.pytorch_ddp_dist_helper.to_ddp_config


registry
========
Please refer to ``ding/utils/registry`` for more details.

Registry
~~~~~~~~
.. autoclass:: ding.utils.registry.Registry
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

render_helper
=============
Please refer to ``ding/utils/render_helper`` for more details.

render_env
~~~~~~~~~~
.. autofunction:: ding.utils.render_helper.render_env

render
~~~~~~~
.. autofunction:: ding.utils.render_helper.render

get_env_fps
~~~~~~~~~~~
.. autofunction:: ding.utils.render_helper.get_env_fps

fps
~~~~~~~
.. autofunction:: ding.utils.render_helper.fps

scheduler_helper
================
Please refer to ``ding/utils/scheduler_helper`` for more details.

Scheduler
~~~~~~~~~
.. autoclass:: ding.utils.scheduler_helper.Scheduler
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

segment_tree
============
Please refer to ``ding/utils/segment_tree`` for more details.

njit
~~~~~~~
.. autofunction:: ding.utils.segment_tree.njit

SegmentTree
~~~~~~~~~~~
.. autoclass:: ding.utils.segment_tree.SegmentTree
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

SumSegmentTree
~~~~~~~~~~~~~~
.. autoclass:: ding.utils.segment_tree.SumSegmentTree
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

MinSegmentTree
~~~~~~~~~~~~~~
.. autoclass:: ding.utils.segment_tree.MinSegmentTree
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

_setitem
~~~~~~~~~
.. autofunction:: ding.utils.segment_tree._setitem

_reduce
~~~~~~~
.. autofunction:: ding.utils.segment_tree._reduce

_find_prefixsum_idx
~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.segment_tree._find_prefixsum_idx

slurm_helper
============
Please refer to ``ding/utils/slurm_helper`` for more details.

get_ip
~~~~~~~
.. autofunction:: ding.utils.slurm_helper.get_ip

get_manager_node_ip
~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.slurm_helper.get_manager_node_ip

get_cls_info
~~~~~~~~~~~~
.. autofunction:: ding.utils.slurm_helper.get_cls_info

node_to_partition
~~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.slurm_helper.node_to_partition

node_to_host
~~~~~~~~~~~~
.. autofunction:: ding.utils.slurm_helper.node_to_host

find_free_port_slurm
~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.slurm_helper.find_free_port_slurm

system_helper
=============
Please refer to ``ding/utils/system_helper`` for more details.

get_ip
~~~~~~~
.. autofunction:: ding.utils.system_helper.get_ip

get_pid
~~~~~~~
.. autofunction:: ding.utils.system_helper.get_pid

get_task_uid
~~~~~~~~~~~~
.. autofunction:: ding.utils.system_helper.get_task_uid

PropagatingThread
~~~~~~~~~~~~~~~~~
.. autoclass:: ding.utils.system_helper.PropagatingThread
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

find_free_port
~~~~~~~~~~~~~~
.. autofunction:: ding.utils.system_helper.find_free_port

time_helper_base
================
Please refer to ``ding/utils/time_helper_base`` for more details.

TimeWrapper
~~~~~~~~~~~
.. autoclass:: ding.utils.time_helper_base.TimeWrapper
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

time_helper_cuda
================
Please refer to ``ding/utils/time_helper_cuda`` for more details.

get_cuda_time_wrapper
~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.time_helper_cuda.get_cuda_time_wrapper

time_helper
===========
Please refer to ``ding/utils/time_helper`` for more details.

build_time_helper
~~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.time_helper.build_time_helper

EasyTimer
~~~~~~~~~
.. autoclass:: ding.utils.time_helper.EasyTimer
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

TimeWrapperTime
~~~~~~~~~~~~~~~
.. autoclass:: ding.utils.time_helper.TimeWrapperTime
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

WatchDog
~~~~~~~~
.. autoclass:: ding.utils.time_helper.WatchDog
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

loader.base
===========
Please refer to ``ding/utils/loader/base`` for more details.

ILoaderClass
~~~~~~~~~~~~
.. autoclass:: ding.utils.loader.base.ILoaderClass
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

loader.collection
=================
Please refer to ``ding/utils/loader/collection`` for more details.

CollectionError
~~~~~~~~~~~~~~~
.. autoclass:: ding.utils.loader.collection.CollectionError
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

collection
~~~~~~~~~~
.. autofunction:: ding.utils.loader.collection.collection

tuple_
~~~~~~~
.. autofunction:: ding.utils.loader.collection.tuple_

length
~~~~~~~
.. autofunction:: ding.utils.loader.collection.length

length_is
~~~~~~~~~
.. autofunction:: ding.utils.loader.collection.length_is

contains
~~~~~~~~
.. autofunction:: ding.utils.loader.collection.contains

cofilter
~~~~~~~~~
.. autofunction:: ding.utils.loader.collection.cofilter

tpselector
~~~~~~~~~~
.. autofunction:: ding.utils.loader.collection.tpselector


loader.dict
===========
Please refer to ``ding/utils/loader/dict`` for more details.

DictError
~~~~~~~~~
.. autoclass:: ding.utils.loader.dict.DictError
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

dict_
~~~~~~~
.. autofunction:: ding.utils.loader.dict.dict_


loader.exception
=================
Please refer to ``ding/utils/loader/exception`` for more details.

CompositeStructureError
~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.utils.loader.exception.CompositeStructureError
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

loader.mapping
==============
Please refer to ``ding/utils/loader/mapping`` for more details.

MappingError
~~~~~~~~~~~~
.. autoclass:: ding.utils.loader.mapping.MappingError
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

mapping
~~~~~~~
.. autofunction:: ding.utils.loader.mapping.mapping

mpfilter
~~~~~~~~
.. autofunction:: ding.utils.loader.mapping.mpfilter

mpkeys
~~~~~~~
.. autofunction:: ding.utils.loader.mapping.mpkeys

mpvalues
~~~~~~~~
.. autofunction:: ding.utils.loader.mapping.mpvalues

mpitems
~~~~~~~
.. autofunction:: ding.utils.loader.mapping.mpitems

item
~~~~~~~
.. autofunction:: ding.utils.loader.mapping.item

item_or
~~~~~~~
.. autofunction:: ding.utils.loader.mapping.item_or

loader.norm
===========
Please refer to ``ding/utils/loader/norm`` for more details.

_callable_to_norm
~~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.loader.norm._callable_to_norm

norm
~~~~~~~
.. autofunction:: ding.utils.loader.norm.norm

normfunc
~~~~~~~~
.. autofunction:: ding.utils.loader.norm.normfunc

_unary
~~~~~~~
.. autofunction:: ding.utils.loader.norm._unary

_binary
~~~~~~~
.. autofunction:: ding.utils.loader.norm._binary

_binary_reducing
~~~~~~~~~~~~~~~~~
.. autofunction:: ding.utils.loader.norm._binary_reducing

INormClass
~~~~~~~~~~
.. autoclass:: ding.utils.loader.norm.INormClass
    :special-members: __init__
    :members:
    :private-members:
    :undoc-members:

lcmp
~~~~~~~
.. autofunction:: ding.utils.loader.norm.lcmp

loader.number
=============
Please refer to ``ding/utils/loader/number`` for more details.

numeric
~~~~~~~
.. autofunction:: ding.utils.loader.number.numeric

interval
~~~~~~~~
.. autofunction:: ding.utils.loader.number.interval

is_negative
~~~~~~~~~~~
.. autofunction:: ding.utils.loader.number.is_negative

is_positive
~~~~~~~~~~~
.. autofunction:: ding.utils.loader.number.is_positive

non_negative
~~~~~~~~~~~~
.. autofunction:: ding.utils.loader.number.non_negative

non_positive
~~~~~~~~~~~~
.. autofunction:: ding.utils.loader.number.non_positive

negative
~~~~~~~~
.. autofunction:: ding.utils.loader.number.negative

positive
~~~~~~~~
.. autofunction:: ding.utils.loader.number.positive

_math_binary
~~~~~~~~~~~~
.. autofunction:: ding.utils.loader.number._math_binary

plus
~~~~~~~
.. autofunction:: ding.utils.loader.number.plus

minus
~~~~~~~
.. autofunction:: ding.utils.loader.number.minus

minus_with
~~~~~~~~~~~
.. autofunction:: ding.utils.loader.number.minus_with

multi
~~~~~~~
.. autofunction:: ding.utils.loader.number.multi

divide
~~~~~~~
.. autofunction:: ding.utils.loader.number.divide

divide_with
~~~~~~~~~~~
.. autofunction:: ding.utils.loader.number.divide_with

power
~~~~~~~
.. autofunction:: ding.utils.loader.number.power

power_with
~~~~~~~~~~~
.. autofunction:: ding.utils.loader.number.power_with

msum
~~~~~~~
.. autofunction:: ding.utils.loader.number.msum

mmulti
~~~~~~~
.. autofunction:: ding.utils.loader.number.mmulti

_msinglecmp
~~~~~~~~~~~
.. autofunction:: ding.utils.loader.number._msinglecmp

mcmp
~~~~~~~
.. autofunction:: ding.utils.loader.number.mcmp

loader.string
=============
Please refer to ``ding/utils/loader/string`` for more details.

enum
~~~~~~~
.. autofunction:: ding.utils.loader.string.enum

_to_regexp
~~~~~~~~~~~
.. autofunction:: ding.utils.loader.string._to_regexp

rematch
~~~~~~~
.. autofunction:: ding.utils.loader.string.rematch

regrep
~~~~~~~
.. autofunction:: ding.utils.loader.string.regrep

loader.types
============
Please refer to ``ding/utils/loader/types`` for more details.

is_type
~~~~~~~
.. autofunction:: ding.utils.loader.types.is_type

to_type
~~~~~~~
.. autofunction:: ding.utils.loader.types.to_type

is_callable
~~~~~~~~~~~~
.. autofunction:: ding.utils.loader.types.is_callable

prop
~~~~~~~
.. autofunction:: ding.utils.loader.types.prop

method
~~~~~~~
.. autofunction:: ding.utils.loader.types.method

fcall
~~~~~~~
.. autofunction:: ding.utils.loader.types.fcall

fpartial
~~~~~~~~
.. autofunction:: ding.utils.loader.types.fpartial

loader.utils
============
Please refer to ``ding/utils/loader/utils`` for more details.

keep
~~~~~~~
.. autofunction:: ding.utils.loader.utils.keep

raw
~~~~~~~
.. autofunction:: ding.utils.loader.utils.raw

optional
~~~~~~~~
.. autofunction:: ding.utils.loader.utils.optional

check_only
~~~~~~~~~~
.. autofunction:: ding.utils.loader.utils.check_only

check
~~~~~~~
.. autofunction:: ding.utils.loader.utils.check
