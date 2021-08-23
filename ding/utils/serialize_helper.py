import time
import numpy as np
import multiprocessing as mp
from ding.torch_utils.data_helper import to_list, to_ndarray
from multiprocessing import Process, Manager, Pipe, connection, get_context, Array, Lock, RawArray
import ctypes
import pyarrow.plasma as plasma
from ding.utils.import_helper import try_import_pyarrow,try_import_pickle



pickle = try_import_pickle()
pyarrow = try_import_pyarrow()

def bold(x):
    return '\033[1m' + str(x) + '\033[0m'


def dim(x):
    return '\033[2m' + str(x) + '\033[0m'


def italicized(x):
    return '\033[3m' + str(x) + '\033[0m'


def underline(x):
    return '\033[4m' + str(x) + '\033[0m'


def blink(x):
    return '\033[5m' + str(x) + '\033[0m'


def inverse(x):
    return '\033[7m' + str(x) + '\033[0m'


def gray(x):
    return '\033[90m' + str(x) + '\033[0m'


def red(x):
    return '\033[91m' + str(x) + '\033[0m'


def green(x):
    return '\033[92m' + str(x) + '\033[0m'


def yellow(x):
    return '\033[93m' + str(x) + '\033[0m'


def blue(x):
    return '\033[94m' + str(x) + '\033[0m'


def magenta(x):
    return '\033[95m' + str(x) + '\033[0m'


def cyan(x):
    return '\033[96m' + str(x) + '\033[0m'


def white(x):
    return '\033[97m' + str(x) + '\033[0m'


class Tick():

    def __init__(self, name='', silent=False):
        self.name = name
        self.silent = silent

    def __enter__(self):
        self.t_start = time.time()
        if not self.silent:
            print('%s ' % (self.name), end='', flush=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.t_end = time.time()
        self.delta = self.t_end - self.t_start
        self.fps = 1 / self.delta

        if not self.silent:
            print(yellow('[%.3fs]' % (self.delta), ), flush=True)


class Tock():

    def __init__(self, name=None, report_time=True):
        self.name = '' if name == None else name + ':'
        self.report_time = report_time

    def __enter__(self):
        self.t_start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.t_end = time.time()
        self.delta = self.t_end - self.t_start
        self.fps = 1 / self.delta
        if self.report_time:
            print(yellow(self.name) + cyan('%.3fs' % (self.delta)), end=' ', flush=True)
        else:
            print(yellow('.'), end='', flush=True)


class MultiprocessingPickle:

    def __init__(self, data, shape, cpu_count, mode='dump', byte_length = -1) -> None:
        self.data = data
        self.shape = shape
        self.cpu_count = cpu_count
        self.mode = mode
        self.byte_length = byte_length

        

    def job_serialized(self, i, buffer, byte_length):
        serialized_x = pickle.dumps(self.arr[i], protocol=pickle.HIGHEST_PROTOCOL)
        buffer[i*byte_length:(i+1)*byte_length] = serialized_x[:]


    def job_deserialized(self, i, buffer, byte_length,split_length):
        deserialized_x = pickle.loads(self.data[i*byte_length:(i+1)*byte_length])
        buffer[i*split_length:(i+1)*split_length] = deserialized_x[:]

    def pack(self):

        self.data = self.data.reshape(-1)
        self.arr = to_ndarray(np.array_split(self.data, self.cpu_count))
        processes = []

        #get byte_length
        if self.byte_length == -1:
            serialized_x_0 = pickle.dumps(self.arr[0], protocol=pickle.HIGHEST_PROTOCOL)
            self.byte_length = len(serialized_x_0)
            buffer = Array(ctypes.c_char, self.cpu_count*self.byte_length)
            buffer[:self.byte_length] = serialized_x_0[:]
            start = 1
            end = self.cpu_count
        else:
            buffer = Array(ctypes.c_char, self.cpu_count*self.byte_length)
            start = 0
            end = self.cpu_count
    
        for i in range(start,end):
            p = mp.Process(target=self.job_serialized, args=(i,buffer, self.byte_length))
            p.start()
            processes.append(p)
        for precess in processes:
            precess.join()
        return buffer[:], self.byte_length

    def unpack(self):
        processes = []
        buffer = mp.Array(ctypes.c_double, int(np.prod(self.shape)))
        assert self.byte_length != -1
        split_length = int(np.prod(self.shape)/self.cpu_count)
        for i in range(self.cpu_count):
            p = mp.Process(target=self.job_deserialized, args=(i,buffer,self.byte_length,split_length))
            p.start()
            processes.append(p)
        for precess in processes:
            precess.join()
        return buffer[:]


class MultiprocessingPyarrow:
    def __init__(self, data, shape, cpu_count, mode='dump', byte_length = -1) -> None:
        self.data = data
        self.shape = shape
        self.cpu_count = cpu_count
        self.mode = mode
        self.byte_length = byte_length
        

    def job_serialized(self, i, buffer, byte_length):
        serialized_x = pyarrow.serialize(self.arr[i]).to_buffer()
        buffer[i*byte_length:(i+1)*byte_length] = serialized_x[:]


    def job_deserialized(self, i, buffer, byte_length,split_length):
        deserialized_x = pyarrow.deserialize(self.data[i*byte_length:(i+1)*byte_length])
        buffer[i*split_length:(i+1)*split_length] = deserialized_x[:]

    def pack(self):

        self.data = self.data.reshape(-1)
        self.arr = to_ndarray(np.array_split(self.data, self.cpu_count))
        processes = []

        #get byte_length
        if self.byte_length == -1:
            serialized_x_0 = pyarrow.serialize(self.arr[0]).to_buffer()
            self.byte_length = len(serialized_x_0)
            buffer = Array(ctypes.c_char, self.cpu_count*self.byte_length)
            buffer[:self.byte_length] = serialized_x_0[:]
            start = 1
            end = self.cpu_count
        else:
            buffer = Array(ctypes.c_char, self.cpu_count*self.byte_length)
            start = 0
            end = self.cpu_count
            
        for i in range(start,end):
            p = mp.Process(target=self.job_serialized, args=(i,buffer, self.byte_length))
            p.start()
            processes.append(p)
        for precess in processes:
            precess.join()
        return buffer[:], self.byte_length

    def unpack(self):
        processes = []
        buffer = mp.Array(ctypes.c_double, int(np.prod(self.shape)))
        assert self.byte_length != -1
        split_length = int(np.prod(self.shape)/self.cpu_count)
        for i in range(self.cpu_count):
            p = mp.Process(target=self.job_deserialized, args=(i,buffer,self.byte_length,split_length))
            p.start()
            processes.append(p)
        for precess in processes:
            precess.join()
        return buffer[:]