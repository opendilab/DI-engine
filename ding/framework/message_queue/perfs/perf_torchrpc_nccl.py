import time
import torch
import os
import argparse
import torch.distributed as dist
import treetensor.torch as ttorch

from dataclasses import dataclass
from queue import Empty
from typing import TYPE_CHECKING, List, Dict, Union
from ditk import logging

from ding.utils.data.structure.lifo_deque import LifoDeque
from ding.framework.message_queue.torch_rpc import DeviceMap, TORCHRPCMQ, RPCEvent
from ding.utils.comm_perf_helper import tensor_size_beauty_print, byte_beauty_print, \
    dtype_2_byte, DO_PERF, time_perf_avg, time_perf_once, print_timer_result_csv

LENGTH = 5
REPEAT = 2
MAX_EXP_NUMS = 10
UNIT_SIZE_LIST = [64, 1024, 64 * 1024, 512 * 1024, 2 * 1024 * 1024]


@dataclass
class SendInfo:
    send_map_dict: Dict = None
    sending_flag: int = 0


# Global vars definition is here:
global mq
global global_send_info_dict
mq = None

global_send_info_dict = dict()


def remote_mq_entrance(topic, *args, **kwargs):
    global mq
    mq.rpc_event_router(topic, *args, **kwargs)


def dict_tensor_send_by_key(
        key: str, tensor_id: int, tensor: torch.Tensor, nums: int, send_id: int, use_cuda: bool
) -> None:
    """
    Overview:
        For data structures that use dict to store tensor, such as dict[key:tensor] or treetensor,
        this function can be used. Each key is transmitted using one rpc, and the rpc transmission
        of each key is asynchronous.
    Arguments:
        - key (str): Key in dict.
        - tensor_id (int): The sending tensor ID during one dict/treetensor rpc transmission.
        - tensor (torch.tensor): The tensor to be sent.
        - nums (int): The total number of sent tensors.
        - send_id (int): The ID of this dict/treetensor rpc transmission.
    """
    global global_send_info_dict
    send_info_dict = global_send_info_dict
    send_info = None

    assert isinstance(key, str)
    assert isinstance(tensor_id, int)
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(nums, int)
    assert isinstance(send_id, int)
    assert isinstance(use_cuda, bool)

    if tensor_id == 0:
        send_info = SendInfo()
        send_info.send_map_dict = dict()
        send_info_dict[send_id] = send_info
    else:
        while True:
            if send_id in send_info_dict.keys():
                send_info = send_info_dict[send_id]
                if send_info is not None:
                    break

    assert isinstance(send_info, SendInfo)

    if key in send_info.send_map_dict.keys():
        raise RuntimeError("Multiple state_dict's key \"{}\" received!".format(key))

    send_info.send_map_dict[key] = tensor

    if tensor_id == nums - 1:
        while len(send_info.send_map_dict) != nums:
            time.sleep(0.01)

        send_info_dict.clear()
        if use_cuda:
            torch.cuda.synchronize(0)
    return


def send_dummy(playload: Union[torch.Tensor, Dict], use_cuda: bool, *args) -> None:
    assert isinstance(use_cuda, bool)
    if use_cuda:
        torch.cuda.synchronize(0)
    return


def dict_tensor_send(mq: TORCHRPCMQ, state_dict: Dict, send_id: int, use_cuda: bool) -> None:
    future_list = []
    for tensor_id, (key, value) in enumerate(state_dict.items()):
        future_list.append(mq.publish("DICT_TENSOR_SEND", key, tensor_id, value, len(state_dict), send_id, use_cuda))

    for future in future_list:
        future.wait()


def perf_torch_rpc(use_cuda=True):
    global LENGTH
    global UNIT_SIZE_LIST
    if use_cuda:
        device = "cuda:0"
    else:
        device = "cpu"

    for i, unit_size in enumerate(UNIT_SIZE_LIST):
        unit_tensor = torch.ones([unit_size * LENGTH]).to(device)
        tensor_dict = {}
        for j in range(LENGTH):
            tensor_dict[str(j)] = torch.ones(unit_size).to(device)

        if use_cuda:
            torch.cuda.synchronize(0)

        @time_perf_avg(1, REPEAT, cuda=use_cuda)
        def one_shot_rpc():
            dict_tensor_send(mq, {'test': unit_tensor}, i, use_cuda)

        @time_perf_avg(1, REPEAT, cuda=use_cuda)
        def one_shot_rpc_with_dict():
            dict_tensor_send(mq, tensor_dict, i, use_cuda)

        @time_perf_avg(1, REPEAT, cuda=use_cuda)
        def split_chunk_rpc():
            re = mq.publish(RPCEvent.CUSTOM_FUNCRION_RPC, {'test': unit_tensor}, use_cuda, custom_method=send_dummy)
            re.wait()

        @time_perf_avg(1, REPEAT, cuda=use_cuda)
        def split_chunk_rpc_with_dict():
            re = mq.publish(RPCEvent.CUSTOM_FUNCRION_RPC, tensor_dict, use_cuda, custom_method=send_dummy)
            re.wait()

        logging.debug("Size {:.2f} {}".format(*byte_beauty_print(unit_size * LENGTH * 4)))

        for idx in range(REPEAT):
            one_shot_rpc(idx)
            one_shot_rpc_with_dict(idx)
            split_chunk_rpc(idx)
            split_chunk_rpc_with_dict(idx)

        if use_cuda:
            torch.cuda.empty_cache()


def perf_nccl(global_rank: int, use_cuda=True):
    if use_cuda:
        device = "cuda:0"
    else:
        device = "cpu"
    ack_tensor = torch.ones(10).to(device)

    if global_rank == 0:
        # Warm up recving
        dist.recv(tensor=ack_tensor, src=1)
        if use_cuda:
            torch.cuda.synchronize(0)

        for i, unit_size in enumerate(UNIT_SIZE_LIST):
            payload = torch.ones([unit_size * LENGTH]).to(device)

            @time_perf_avg(1, REPEAT, cuda=True)
            def test_case_nccl(payload):
                dist.send(tensor=payload, dst=1, tag=i)

            logging.debug("Size {:.2f} {}".format(*byte_beauty_print(unit_size * LENGTH * 4)))

            for idx in range(REPEAT):
                test_case_nccl(idx, payload)
    else:
        # Warm up sending
        dist.send(tensor=ack_tensor, dst=0)
        if use_cuda:
            torch.cuda.synchronize(0)

        for i, unit_size in enumerate(UNIT_SIZE_LIST):
            recvbuffer = torch.ones([unit_size * LENGTH]).to(device)
            for j in range(REPEAT):
                dist.recv(tensor=recvbuffer, src=0, tag=i)
                if use_cuda:
                    torch.cuda.synchronize(0)


def rpc_model_exchanger(rank: int, init_method: str, test_nccl: bool = False, use_cuda: bool = True):
    global mq
    global dict_tensor_send_by_key
    global remote_mq_entrance
    from ding.framework.parallel import Parallel

    logging.getLogger().setLevel(logging.DEBUG)
    if test_nccl:
        dist.init_process_group("nccl", rank=rank, world_size=2, init_method=init_method)
    params = Parallel._torchrpc_args_parser(
        n_parallel_workers=1,
        attach_to=[1] if rank == 0 else [],
        node_ids=[rank],
        init_method=init_method,
        use_cuda=use_cuda,
        async_rpc=True,
        async_backend_polling=False,
        remote_parallel_entrance=remote_mq_entrance
    )[0]
    logging.debug(params)
    mq = TORCHRPCMQ(**params)
    mq.show_device_maps()

    # Because the dict_tensor_send_by_key() relies on global variables, we have to register it.
    mq.subscribe("DICT_TENSOR_SEND", dict_tensor_send_by_key)
    mq.listen()

    # In order to prevent deadlock caused by mixed use of "torch.cuda.synchronize" between
    # nccl and torchrpc, we test the two backend separately.
    if rank == 1:
        # Receiver ready for testing nccl
        if test_nccl:
            perf_nccl(rank)
        # Receiver join to wait sender to send shutdown signal.
        mq.wait_for_shutdown()
    elif rank == 0:
        # Sender test torch rpc.
        perf_torch_rpc(use_cuda=use_cuda)
        # Sender test nccl.
        if test_nccl:
            perf_nccl(rank)
        # Print test results.
        print_timer_result_csv()
        # Sender send finish signal.
        mq.require_to_shutdown("Node_1")
        # Sender clean resources.
        mq.stop()


# Usage:
# CUDA_VISIBLE_DEVICES=0 python perf_torchrpc_nccl.py --rank=0
# CUDA_VISIBLE_DEVICES=1 python perf_torchrpc_nccl.py --rank=1
#
# Note:
# If you are in a container, please ensure that your /dev/shm is large enough.
# If there is a strange core or bug, please check if /dev/shm is full.
# If so, please try to clear it manually:
# /dev/shm/nccl*
# /dev/shm/cuda.shm.*
# /dev/shm/torch_*
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test torch rpc')
    parser.add_argument('--rank', type=int)
    parser.add_argument('--init-method', type=str, default="tcp://127.0.0.1:12347")
    parser.add_argument('--test_nccl', type=bool, default=False)
    parser.add_argument('--use_cuda', type=bool, default=False)
    args, _ = parser.parse_known_args()

    if args.use_cuda:
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            logging.info("CUDA_VISIBLE_DEVICES: {}".format(os.environ['CUDA_VISIBLE_DEVICES']))
        else:
            logging.info("Not set CUDA_VISIBLE_DEVICES!")

        logging.info(
            "CUDA is enable:{}, nums of GPU: {}, current device: {}".format(
                torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.current_device()
            )
        )

    rpc_model_exchanger(args.rank, args.init_method, args.test_nccl, args.use_cuda)
