import time
import numpy as np
import multiprocessing as mp
from ding.torch_utils.data_helper import to_list, to_ndarray
from multiprocessing import Process, Manager, Pipe, connection, get_context, Array, Lock, RawArray
import ctypes
import pyarrow.plasma as plasma
from ding.utils.import_helper import try_import_pyarrow, try_import_pickle
from ding.utils.default_helper import yellow, cyan, Tick, Tock

pickle = try_import_pickle()
pyarrow = try_import_pyarrow()


class MultiprocessingPickle:

    def __init__(self, data, shape, cpu_count, mode='dump', byte_length=-1) -> None:
        self.data = data
        self.shape = shape
        self.cpu_count = cpu_count
        self.mode = mode
        self.byte_length = byte_length

    def job_serialized(self, i, buffer, byte_length):
        serialized_x = pickle.dumps(self.arr[i], protocol=pickle.HIGHEST_PROTOCOL)
        buffer[i * byte_length:(i + 1) * byte_length] = serialized_x[:]

    def job_deserialized(self, i, buffer, byte_length, split_length):
        deserialized_x = pickle.loads(self.data[i * byte_length:(i + 1) * byte_length])
        buffer[i * split_length:(i + 1) * split_length] = deserialized_x[:]

    def pack(self):

        self.data = self.data.reshape(-1)
        self.arr = to_ndarray(np.array_split(self.data, self.cpu_count))
        processes = []

        #get byte_length
        if self.byte_length == -1:
            serialized_x_0 = pickle.dumps(self.arr[0], protocol=pickle.HIGHEST_PROTOCOL)
            self.byte_length = len(serialized_x_0)
            buffer = Array(ctypes.c_char, self.cpu_count * self.byte_length)
            buffer[:self.byte_length] = serialized_x_0[:]
            start = 1
            end = self.cpu_count
        else:
            buffer = Array(ctypes.c_char, self.cpu_count * self.byte_length)
            start = 0
            end = self.cpu_count

        for i in range(start, end):
            p = mp.Process(target=self.job_serialized, args=(i, buffer, self.byte_length))
            p.start()
            processes.append(p)
        for precess in processes:
            precess.join()
        return buffer[:], self.byte_length

    def unpack(self):
        processes = []
        buffer = mp.Array(ctypes.c_double, int(np.prod(self.shape)))
        assert self.byte_length != -1
        split_length = int(np.prod(self.shape) / self.cpu_count)
        for i in range(self.cpu_count):
            p = mp.Process(target=self.job_deserialized, args=(i, buffer, self.byte_length, split_length))
            p.start()
            processes.append(p)
        for precess in processes:
            precess.join()
        return buffer[:]


class MultiprocessingPyarrow:

    def __init__(self, data, shape, cpu_count, mode='dump', byte_length=-1) -> None:
        self.data = data
        self.shape = shape
        self.cpu_count = cpu_count
        self.mode = mode
        self.byte_length = byte_length

    def job_serialized(self, i, buffer, byte_length):
        serialized_x = pyarrow.serialize(self.arr[i]).to_buffer()
        buffer[i * byte_length:(i + 1) * byte_length] = serialized_x[:]

    def job_deserialized(self, i, buffer, byte_length, split_length):
        deserialized_x = pyarrow.deserialize(self.data[i * byte_length:(i + 1) * byte_length])
        buffer[i * split_length:(i + 1) * split_length] = deserialized_x[:]

    def pack(self):

        self.data = self.data.reshape(-1)
        self.arr = to_ndarray(np.array_split(self.data, self.cpu_count))
        processes = []

        #get byte_length
        if self.byte_length == -1:
            serialized_x_0 = pyarrow.serialize(self.arr[0]).to_buffer()
            self.byte_length = len(serialized_x_0)
            buffer = Array(ctypes.c_char, self.cpu_count * self.byte_length)
            buffer[:self.byte_length] = serialized_x_0[:]
            start = 1
            end = self.cpu_count
        else:
            buffer = Array(ctypes.c_char, self.cpu_count * self.byte_length)
            start = 0
            end = self.cpu_count

        for i in range(start, end):
            p = mp.Process(target=self.job_serialized, args=(i, buffer, self.byte_length))
            p.start()
            processes.append(p)
        for precess in processes:
            precess.join()
        return buffer[:], self.byte_length

    def unpack(self):
        processes = []
        buffer = mp.Array(ctypes.c_double, int(np.prod(self.shape)))
        assert self.byte_length != -1
        split_length = int(np.prod(self.shape) / self.cpu_count)
        for i in range(self.cpu_count):
            p = mp.Process(target=self.job_deserialized, args=(i, buffer, self.byte_length, split_length))
            p.start()
            processes.append(p)
        for precess in processes:
            precess.join()
        return buffer[:]
