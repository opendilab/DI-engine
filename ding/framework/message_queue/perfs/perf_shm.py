from typing import TYPE_CHECKING, Any, List, Union, Dict, Optional, Callable

from ditk import logging
from ding.framework.supervisor import RecvPayload, SendPayload, Supervisor, ChildType
from ding.envs.env_manager.subprocess_env_manager import ShmBufferContainer, ShmBuffer
from ding.utils.comm_perf_helper import tensor_size_beauty_print, byte_beauty_print, \
    dtype_2_byte, TENSOR_SIZE_LIST, print_timer_result_csv

import torch
import numpy as np
import time
import argparse

LENGTH = 5
REPEAT = 10
UNIT_SIZE_LIST = [8, 64, 1024, 64 * 1024, 512 * 1024, 2 * 1024 * 1024, 32 * 1024 * 1024, 64 * 1024 * 1024]
logging.getLogger().setLevel(logging.INFO)


def shm_callback(payload: RecvPayload, buffers: Any):
    # Step4: shared memory -> np.array
    np_tensor = buffers[payload.data["idx"]].get()
    # Step5: np.array -> cpu tensor
    tensor = torch.from_numpy(np_tensor)
    # Step6: cpu tensor -> gpu tensor
    tensor = tensor.cuda(0)
    torch.cuda.synchronize(0)


def cuda_shm_callback(payload: RecvPayload, buffers: Any):
    # Step2: gpu shared tensor -> gpu tensor
    tensor = buffers[payload.data["idx"]].get()
    assert tensor.device == torch.device('cuda:0')
    # Step3: gpu tensor(cuda:0) -> gpu tensor(cuda:1)
    tensor = tensor.to(1)
    torch.cuda.synchronize(1)
    assert tensor.device == torch.device('cuda:1')


class Recvier:

    def step(self, idx: int, __start_time):
        return {"idx": idx, "start_time": __start_time}


class ShmSupervisor(Supervisor):

    def __init__(self, gpu_tensors, buffers, ctx, is_cuda_buffer):
        super().__init__(type_=ChildType.PROCESS, mp_ctx=ctx)
        self.gpu_tensors = gpu_tensors
        self.buffers = buffers
        self.time_list = []
        self._time_list = []
        self._is_cuda_buffer = is_cuda_buffer
        if not is_cuda_buffer:
            _shm_callback = shm_callback
        else:
            _shm_callback = cuda_shm_callback
        self.register(Recvier, shm_buffer=self.buffers, shm_callback=_shm_callback)
        super().start_link()

    def _send_recv_callback(self, payload: RecvPayload, remain_payloads: Optional[Dict[str, SendPayload]] = None):
        idx = payload.data["idx"]
        __start_time = payload.data["start_time"]
        __end_time = time.time()
        self.time_list.append(float(__end_time - __start_time) * 1000.0)

    def step(self):
        # Do not use Queue to send large data, use shm.
        for i, size in enumerate(UNIT_SIZE_LIST):
            for j in range(REPEAT):
                __start_time = time.time()

                if not self._is_cuda_buffer:
                    # Numpy shm buffer:
                    # Step1: gpu tensor -> cpu tensor
                    tensor = self.gpu_tensors[i].cpu()
                    # Step2: cpu tensor-> np.array
                    np_tensor = tensor.numpy()
                    # Step3: np.array -> shared memory
                    self.buffers[i].fill(np_tensor)
                else:
                    # Cuda shared tensor
                    # Step1: gpu tensor -> gpu shared tensor
                    self.buffers[i].fill(self.gpu_tensors[i])

                payload = SendPayload(proc_id=0, method="step", args=[i, __start_time])
                send_payloads = [payload]

                self.send(payload)
                self.recv_all(send_payloads, ignore_err=True, callback=self._send_recv_callback)

            _avg_time = sum(self.time_list) / len(self.time_list)
            self._time_list.append(_avg_time)
            self.time_list.clear()
            logging.info(
                "Data size {:.2f} {} , repeat {}, avg RTT {:.4f} ms".format(
                    *byte_beauty_print(UNIT_SIZE_LIST[i] * 4 * LENGTH), REPEAT, _avg_time
                )
            )

        for t in self._time_list:
            print("{:.4f},".format(t), end="")
        print("")


def shm_perf_main(test_type: str):
    gpu_tensors = list()
    buffers = dict()

    if test_type == "shm":
        import multiprocessing as mp
        use_cuda_buffer = False
    elif test_type == "cuda_ipc":
        use_cuda_buffer = True
        import torch.multiprocessing as mp

    ctx = mp.get_context('spawn')

    for i, size in enumerate(UNIT_SIZE_LIST):
        unit_size = size * LENGTH
        gpu_tensors.append(torch.ones(unit_size).cuda(0))
        if not use_cuda_buffer:
            buffers[i] = ShmBufferContainer(np.float32, (unit_size, ), copy_on_get=True, is_cuda_buffer=False)
        else:
            buffers[i] = ShmBufferContainer(torch.float32, (unit_size, ), copy_on_get=True, is_cuda_buffer=True)

    sv = ShmSupervisor(
        gpu_tensors=gpu_tensors, buffers=buffers, ctx=mp.get_context('spawn'), is_cuda_buffer=use_cuda_buffer
    )
    sv.step()
    del sv


# Usages:
# python perf_shm.py --test_type ["shm"|"cuda_ipc"]
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test torch rpc')
    parser.add_argument('--test_type', type=str)
    args, _ = parser.parse_known_args()
    shm_perf_main(args.test_type)
