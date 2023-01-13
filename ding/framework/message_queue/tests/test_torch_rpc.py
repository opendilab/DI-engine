import pytest
import torch
import platform
import time
import socket

from ding.framework.message_queue.torch_rpc import DeviceMap, TORCHRPCMQ, DEFAULT_DEVICE_MAP_NUMS
from torch.distributed import rpc
from multiprocessing import Pool, get_context
from ding.compatibility import torch_ge_1121
from ditk import logging
from ding.utils.system_helper import find_free_port

mq = None
recv_tensor_list = [None, None, None, None]


def remote_mq_entrance(topic, *args, **kwargs):
    global mq
    mq.rpc_event_router(topic, *args, **kwargs)


def torchrpc(rank):
    global mq
    global recv_tensor_list
    mq = None
    address = socket.gethostbyname(socket.gethostname())
    recv_tensor_list = [None, None, None, None]
    logging.getLogger().setLevel(logging.DEBUG)
    name_list = ["A", "B", "C", "D"]

    if rank == 0:
        attach_to = name_list[1:]
    else:
        attach_to = None

    mq = TORCHRPCMQ(
        rpc_name=name_list[rank],
        global_rank=rank,
        init_method="tcp://{}:12398".format(address),
        remote_parallel_entrance=remote_mq_entrance,
        attach_to=attach_to,
        async_rpc=False,
        use_cuda=False
    )

    def fn1(tensor: torch.Tensor) -> None:
        global recv_tensor_list
        global mq
        recv_tensor_list[0] = tensor
        assert recv_tensor_list[0].sum().item() == 1000
        mq.publish("RANK_N_SEND", torch.ones(10), mq.global_rank)

    def fn2(tensor: torch.Tensor, rank) -> None:
        global recv_tensor_list
        recv_tensor_list[rank] = tensor
        assert recv_tensor_list[rank].sum().item() == 10

    mq.subscribe(topic="RANK_0_SEND", fn=fn1)
    mq.subscribe(topic="RANK_N_SEND", fn=fn2)
    mq.listen()

    if rank == 0:
        mq.publish("RANK_0_SEND", torch.ones(1000))

    mq._rendezvous_until_world_size(4)
    all_worker_info = rpc._get_current_rpc_agent().get_worker_infos()
    rpc.api._barrier([worker.name for worker in all_worker_info])

    mq.unsubscribe("RANK_0_SEND")
    assert "RANK_0_SEND" not in mq._rpc_events

    if rank == 0:
        mq.publish("RANK_0_SEND", torch.ones(1000))

    mq._rendezvous_until_world_size(4)
    rpc.api._barrier(name_list)
    mq.stop()


def torchrpc_cuda(rank):
    global mq
    global recv_tensor_list
    mq = None
    recv_tensor_list = [None, None, None, None]
    name_list = ["A", "B"]
    address = socket.gethostbyname(socket.gethostname())
    logging.getLogger().setLevel(logging.DEBUG)

    if rank == 0:
        attach_to = name_list[1:]
    else:
        attach_to = None

    peer_rank = int(rank == 0) or 0
    peer_name = name_list[peer_rank]
    device_map = DeviceMap(rank, [peer_name], [rank], [peer_rank])
    logging.debug(device_map)

    mq = TORCHRPCMQ(
        rpc_name=name_list[rank],
        global_rank=rank,
        init_method="tcp://{}:12390".format(address),
        remote_parallel_entrance=remote_mq_entrance,
        attach_to=attach_to,
        device_maps=device_map,
        async_rpc=False,
        cuda_device=rank,
        use_cuda=True
    )

    def fn1(tensor: torch.Tensor) -> None:
        global recv_tensor_list
        global mq
        recv_tensor_list[0] = tensor
        assert recv_tensor_list[0].sum().item() == 777
        assert recv_tensor_list[0].device == torch.device(1)

    mq.subscribe(topic="RANK_0_SEND", fn=fn1)
    mq.listen()

    if rank == 0:
        mq.publish("RANK_0_SEND", torch.ones(777).cuda(0))

    mq._rendezvous_until_world_size(2)
    all_worker_info = rpc._get_current_rpc_agent().get_worker_infos()
    rpc.api._barrier([worker.name for worker in all_worker_info])
    mq.stop()


def torchrpc_args_parser(rank):
    global mq
    global recv_tensor_list
    from ding.framework.parallel import Parallel
    logging.getLogger().setLevel(logging.DEBUG)

    params = Parallel._torchrpc_args_parser(
        n_parallel_workers=1,
        attach_to=[],
        node_ids=[0],
        init_method="tcp://127.0.0.1:12399",
        use_cuda=True,
        local_cuda_devices=None,
        cuda_device_map=None
    )[0]

    logging.debug(params)

    # 1. If attach_to is empty, init_rpc will not block.
    mq = TORCHRPCMQ(**params)
    mq.listen()
    assert mq._running
    mq.stop()
    assert not mq._running
    logging.debug("[Pass] 1. If attach_to is empty, init_rpc will not block.")

    # 2. n_parallel_workers != len(node_ids)
    try:
        Parallel._torchrpc_args_parser(n_parallel_workers=999, attach_to=[], node_ids=[1, 2])[0]
    except RuntimeError as e:
        logging.debug("[Pass] 2. n_parallel_workers != len(node_ids).")
    else:
        assert False

    # 3. len(local_cuda_devices) != n_parallel_workers
    try:
        Parallel._torchrpc_args_parser(n_parallel_workers=8, node_ids=[1], local_cuda_devices=[1, 2, 3])[0]
    except RuntimeError as e:
        logging.debug("[Pass] 3. len(local_cuda_devices) != n_parallel_workers.")
    else:
        assert False

    # 4. n_parallel_workers > gpu_nums
    # TODO(wgt): Support spwan mode to start torchrpc process using CPU/CUDA and CPU only.
    try:
        Parallel._torchrpc_args_parser(n_parallel_workers=999, node_ids=[1], use_cuda=True)[0]
    except RuntimeError as e:
        logging.debug("[Pass] 4. n_parallel_workers > gpu_nums.")
    else:
        assert False

    # 5. Set custom device map.
    params = Parallel._torchrpc_args_parser(
        n_parallel_workers=1, node_ids=[1], cuda_device_map=["0_0_0", "0_1_2", "1_1_4"]
    )[0]
    assert params['device_maps'].peer_name_list == ["Node_0", "Node_0", "Node_1"]
    assert params['device_maps'].our_device_list == [0, 1, 1]
    assert params['device_maps'].peer_device_list == [0, 2, 4]
    # logging.debug(params['device_maps'])
    logging.debug("[Pass] 5. Set custom device map.")

    # 6. Set n_parallel_workers > 1
    params = Parallel._torchrpc_args_parser(n_parallel_workers=8, node_ids=[1])
    assert len(params) == 8
    assert params[7]['node_id'] == 8
    assert params[0]['use_cuda'] is False
    assert params[0]['device_maps'] is None
    assert params[0]['cuda_device'] is None

    if torch.cuda.device_count() >= 2:
        params = Parallel._torchrpc_args_parser(n_parallel_workers=2, node_ids=[1], use_cuda=True)
        assert params[0]['use_cuda']
        assert len(params[0]['device_maps'].peer_name_list) == DEFAULT_DEVICE_MAP_NUMS - 1
    logging.debug("[Pass] 6. Set n_parallel_workers > 1.")


@pytest.mark.unittest
def test_torchrpc():
    ctx = get_context("spawn")
    if platform.system().lower() != 'windows' and torch_ge_1121():
        with ctx.Pool(processes=4) as pool:
            pool.map(torchrpc, range(4))
            pool.close()
            pool.join()


@pytest.mark.cudatest
@pytest.mark.unittest
def test_torchrpc_cuda():
    if platform.system().lower() != 'windows':
        if torch_ge_1121() and torch.cuda.is_available() and torch.cuda.device_count() >= 2:
            ctx = get_context("spawn")
            with ctx.Pool(processes=2) as pool:
                pool.map(torchrpc_cuda, range(2))
                pool.close()
                pool.join()


@pytest.mark.cudatest
@pytest.mark.unittest
def test_torchrpc_parser():
    if platform.system().lower() != 'windows' and torch_ge_1121() and torch.cuda.is_available():
        ctx = get_context("spawn")
        with ctx.Pool(processes=1) as pool:
            pool.map(torchrpc_args_parser, range(1))
            pool.close()
            pool.join()
