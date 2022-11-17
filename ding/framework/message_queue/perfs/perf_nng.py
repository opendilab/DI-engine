import pickle
import multiprocessing as mp
import argparse
import os
import time
import torch
import numpy as np
import click
import struct

from time import sleep
from threading import Thread
from ding.framework.message_queue.nng import NNGMQ
from ditk import logging
from ding.framework.parallel import Parallel
from ding.utils.comm_perf_helper import byte_beauty_print, time_perf_avg, print_timer_result_csv
from ding.utils import EasyTimer, WatchDog

logging.getLogger().setLevel(logging.INFO)
REPEAT = 10
LENGTH = 5
EXP_NUMS = 2
UNIT_SIZE_LIST = [64, 1024, 64 * 1024, 512 * 1024, 2 * 1024 * 1024]


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--ports", type=str, default="50515")
@click.option("--attach-to", type=str, help="The addresses to connect to.")
@click.option("--address", type=str, help="The address to listen to (without port).")
@click.option("--labels", type=str, help="Labels.")
@click.option("--node-ids", type=str, help="Candidate node ids.")
def handle_args(*args, **kwargs):
    return nng_perf_main(*args, **kwargs)


def pack_time(data, value):
    if value:
        return struct.pack('d', value) + "::".encode() + data
    else:
        return struct.pack('d', value)


def unpack_time(value):
    return struct.unpack('=d', value)[0]


def nng_dist_main(labels, node_id, listen_to, attach_to, *arg, **kwargs) -> None:
    """
    Overview:
        Since nng message reception may be out of order, and nng
        does not have a handshake, the sender may start
        sending messages and timing before the receiver is ready.
        So this function does the corresponding work.
    """
    mq = NNGMQ(listen_to=listen_to, attach_to=attach_to)
    mq.listen()
    label = labels.pop()
    rank = 0
    future_dict = dict()
    start_tag = []
    finish_tag = []

    def send_t(topic, data=None):
        try:
            if not data:
                data = [0, 0]
            data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            mq.publish(topic, data)
            logging.debug("send topic {}".format(topic))
        except Exception as e:
            logging.error("send error at rank:{} label:\"{}\", topic:\"{}\", error: {}".format(rank, label, topic, e))

    def recv_loop():
        while True:
            topic, data = mq.recv()
            if topic == "z":
                # perf_nng_detail recv callback.
                timestamps, data = data.split(b"::", maxsplit=1)
                h2d_timer = EasyTimer(cuda=True)
                pickle_timer = EasyTimer(cuda=False)

                with pickle_timer:
                    data = pickle.loads(data)
                    data, idx = data[0], data[1]

                with h2d_timer:
                    data = data.cuda(0)

                data = pickle.dumps([timestamps, idx], protocol=pickle.HIGHEST_PROTOCOL)
                time_res = pack_time(data, pickle_timer.value)
                time_res = pack_time(time_res, h2d_timer.value)

                mq.publish("k", time_res)
                continue
            elif topic == "k":
                # perf_nng_detail send callback.
                h2d_time, pickle_time, data = data.split(b"::", maxsplit=2)
                data = pickle.loads(data)
                timestamps, idx = data[0], data[1]
                future_dict['perf_finsh'] = (unpack_time(h2d_time), unpack_time(pickle_time), unpack_time(timestamps))
                future_dict[idx] = 1
                continue
            else:
                # Callback functions for other tests.
                data = pickle.loads(data)
                data, idx = data[0], data[1]
                if topic == "t":
                    assert isinstance(data, torch.Tensor)
                    data = data.cuda(0)
                    torch.cuda.synchronize(0)
                    pass
                elif topic == "d":
                    assert isinstance(data, dict)
                    for k, v in data.items():
                        data[k] = v.cuda(0)
                    torch.cuda.synchronize(0)
                elif topic == "a":
                    if idx not in future_dict.keys():
                        raise RuntimeError("Unkown idx")
                    future_dict[idx] = 1
                    continue
                elif topic == "s":
                    if label == 'collector':
                        send_t("s")
                    elif label == 'learner':
                        start_tag.append(1)
                    continue
                elif topic == "f":
                    finish_tag.append(1)
                    return
                else:
                    raise RuntimeError("Unkown topic")

                send_t("a", ["", idx])

    def irendezvous():
        timeout_killer = WatchDog(3)
        timeout_killer.start()
        send_t("s")
        while len(start_tag) == 0:
            time.sleep(0.05)
        timeout_killer.stop()

    listen_thread = Thread(target=recv_loop, name="recv_loop", daemon=True)
    listen_thread.start()

    if label == 'learner':
        while True:
            try:
                irendezvous()
            except Exception as e:
                logging.warning("timeout for irendezvous")
            else:
                break

    if label == 'learner':

        for size in UNIT_SIZE_LIST:
            unit_size = size * LENGTH
            gpu_data = torch.ones(unit_size).cuda(rank)
            time_list = [list() for i in range(EXP_NUMS)]
            size_lists = [[size] for i in range(LENGTH)]
            send_func_list = []
            logging.info("Data size: {:.2f} {}".format(*byte_beauty_print(unit_size * 4)))
            tensor_dict = dict()
            for j, size_list in enumerate(size_lists):
                tensor_dict[str(j)] = torch.ones(size_list).cuda(rank)

            @time_perf_avg(1, REPEAT, cuda=True)
            def nng_tensor_sender_1(idx):
                future_dict[idx] = 0
                send_t("t", [gpu_data.cpu(), idx])
                while future_dict[idx] == 0:
                    time.sleep(0.03)

            @time_perf_avg(1, REPEAT, cuda=True)
            def nng_tensor_sender_2(idx):
                tmp_dict = dict()
                future_dict[idx] = 0
                for key, value in tensor_dict.items():
                    tmp_dict[key] = value.cpu()
                send_t("d", [tmp_dict, idx])
                while future_dict[idx] == 0:
                    time.sleep(0.03)

            def perf_nng_detail(idx):
                future_dict[idx] = 0
                h2d_timer = EasyTimer(cuda=True)
                pickle_timer = EasyTimer(cuda=False)

                with h2d_timer:
                    data = gpu_data.cpu()

                with pickle_timer:
                    data = pickle.dumps([data, idx], protocol=pickle.HIGHEST_PROTOCOL)

                data = pack_time(data, time.time())
                mq.publish("z", data)

                while future_dict[idx] == 0:
                    time.sleep(0.03)

                peer_h2d_time, peer_pickle_time, timestamps = future_dict['perf_finsh']
                total_time = time.time() - timestamps
                # Serialization time
                pickle_time = peer_pickle_time + pickle_timer.value
                # H2D/D2H time
                pcie_time = peer_h2d_time + h2d_timer.value
                # TCP I/O time
                IO_time = total_time - pickle_time - pcie_time
                logging.info(
                    "Detailed: total:[{:.4f}]ms, pickle:[{:.4f}]ms, H2D/D2H:[{:.4f}]ms, I/O:[{:.4f}]ms".format(
                        total_time, pickle_time, pcie_time, IO_time
                    )
                )
                # print("{:.4f}, {:.4f}, {:.4f}, {:.4f}".format(total_time, pickle_time, pcie_time, IO_time))

            send_func_list.append(nng_tensor_sender_1)
            send_func_list.append(nng_tensor_sender_2)

            for i in range(len(send_func_list)):
                for j in range(REPEAT):
                    send_func_list[i](j, i + j)

            # Determine the time-consuming of each stage of nng.
            perf_nng_detail(0)

            # Do some proper cleanup to prevent cuda memory overflow
            torch.cuda.empty_cache()

    if label == 'learner':
        send_t("f")
        finish_tag.append(1)

    while len(finish_tag) == 0:
        time.sleep(0.1)

    print_timer_result_csv()


def nng_perf_main(ports: str, attach_to: str, address: str, labels: str, node_ids: str):
    if not isinstance(ports, int):
        ports = ports.split(",")
        ports = list(map(lambda i: int(i), ports))
        ports = ports[0] if len(ports) == 1 else ports
    if attach_to:
        attach_to = attach_to.split(",")
        attach_to = list(map(lambda s: s.strip(), attach_to))
    if labels:
        labels = labels.split(",")
        labels = set(map(lambda s: s.strip(), labels))
    if node_ids and not isinstance(node_ids, int):
        node_ids = node_ids.split(",")
        node_ids = list(map(lambda i: int(i), node_ids))

    runner_params = Parallel._nng_args_parser(
        n_parallel_workers=1,
        ports=ports,
        protocol="tcp",
        attach_to=attach_to,
        address=address,
        labels=labels,
        node_ids=node_ids,
    )
    logging.debug(runner_params)
    nng_dist_main(**runner_params[0])


# Usages:
# CUDA_VISIBLE_DEVICES=0 python perf_nng.py --node-ids 0 --labels learner --ports 12345 --address 0.0.0.0
# CUDA_VISIBLE_DEVICES=1 python perf_nng.py --node-ids 1 --labels collector --address 127.0.0.1 \
# --ports 12355 --attach-to tcp://0.0.0.0:12345
if __name__ == "__main__":
    handle_args()
