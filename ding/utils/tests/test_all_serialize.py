#from typing import Protocol
import numpy as np
import pickle
import multiprocessing as mp
import multiprocessing.dummy as dmp
import torch
from ding.torch_utils.data_helper import to_list, to_ndarray
from multiprocessing import Process, Manager, Pipe, connection, get_context, Array
import ctypes
import pyarrow
import subprocess
import pyarrow.plasma as plasma
import datetime
import time

######################
# core 都为6
# share memory  瓶颈在2.5G
# parrow 全程好
# mp_pyarrow 好于 mp_pickle，但受share memory瓶颈影响
# mp_pickle 300M ~ 2.5G 多进程表现好于 pickle
######################

########################################################################################################################################
#                       pickle       pyarrow           mp_pickle             mp_pyarrow       |    mp_pickle+plasma    mp_pyarrow+plasma
# data_100M             0.128        0.067             0.184(0.124)        0.163(0.126)       |      0.138(0.072)        0.114(0.070)
# data_150M             0.124        0.067             0.182(0.122)        0.165(0.128)       |      0.138(0.073)        0.116(0.073)
# data_200M             0.327        0.120             0.333(0.226)        0.274(0.215)       |      0.255(0.126)        0.194(0.125)
# data_250M             0.396        0.140             0.401(0.265)        0.339(0.270)       |      0.300(0.148)        0.232(0.151)
# data_300M             0.505        0.176             0.511(0.332)        0.410(0.328)       |      0.351(0.173)        0.284(0.179)
# data_400M             0.673        0.225             0.662(0.448)        0.544(0.444)       |      0.457(0.2223)       0.334(0.222)
# data_500M             0.802        0.266             0.759(0.506)        0.628(0.507)       |      0.561(0.274)        0.408(0.273)
# data_1G               1.679        0.553             1.640(1.112)        1.266(1.029)       |      1.111(0.503)        0.831(0.553)
# data_2G               3.216        1.051             3.044(2.060)        2.444(1.988)       |      2.185(0.983)        1.504(0.993)
# data_2dot5G           4.348        1.304             9.510(8.278)        9.012(8.443)       |      2.545(0.960)        33.305(32.673)
# data_3G               4.830        1.541             19.368(17.27)       18.568(17.706)     |


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




class Serialized_Zoo():

    def __init__(self, data) -> None:
        self.data = data

    def pickle_cost(self):
        with Tick('pickle'):
            with Tock('dump'):
                tmp = pickle.dumps(self.data, protocol=pickle.HIGHEST_PROTOCOL)
            with Tock('load'):
                recon = pickle.loads(tmp)

    def pyarrow_cost(self):
        with Tick('pyarrow'):
            with Tock('dump'):
                tmp = pyarrow.serialize(self.data).to_buffer()
                print(tmp)
            with Tock('load'):
                recon = pyarrow.deserialize(tmp)
                print(recon)

    def job_1(self, i):
        with Tock('dump-{}'.format(i)):
            serialized_x = pickle.dumps(self.arr[i], protocol=pickle.HIGHEST_PROTOCOL)
        with Tock('load-{}'.format(i)):
            deserialized_x = pickle.loads(serialized_x)

    def job_2(self, i):
        with Tock('dump-{}'.format(i)):
            serialized_x = pyarrow.serialize(self.arr[i]).to_buffer()
        with Tock('load-{}'.format(i)):
            deserialized_x = pyarrow.deserialize(serialized_x)

    def job_3(self, tmp, i):
        with Tock('dump-{}'.format(i)):
            serialized_x = pyarrow.serialize(tmp[i]).to_buffer()
        with Tock('load-{}'.format(i)):
            deserialized_x = pyarrow.deserialize(serialized_x)

    def job_4(self, tmp, i):
        with Tock('dump-{}'.format(i)):
            serialized_x = pickle.dumps(tmp[i], protocol=pickle.HIGHEST_PROTOCOL)
            #print(serialized_x)
        with Tock('load-{}'.format(i)):
            deserialized_x = pickle.loads(serialized_x)
            #print(deserialized_x)

    def build_share_memory(self):
        # self.buffer = Array(ctypes.c_float,int(np.prod(self.data.shape)))
        # self.dst_arr = np.frombuffer(self.buffer.get_obj(), dtype=np.float32).reshape(self.data.shape)
        # with self.buffer.get_lock():
        #     np.copyto(self.dst_arr, self.data)
        self.buffer = Array(ctypes.c_float, int(np.prod(self.data.shape)))
        self.data = self.data.reshape(-1)
        self.dst_arr = np.frombuffer(self.buffer.get_obj(), dtype=np.float32)
        with self.buffer.get_lock():
            np.copyto(self.dst_arr, self.data)

    def get_data_from_share_memory(self, cpu_count):
        #self.arr = np.frombuffer(self.buffer.get_obj(), dtype=np.float32).reshape(self.data.shape)
        self.arr = np.frombuffer(self.buffer.get_obj(), dtype=np.float32)
        self.arr = to_ndarray(np.array_split(self.arr, cpu_count))

    def mp_cost_pickle(self, cpu_count):
        with Tick('mp_cost_pickle'):
            # with Tock('build share memory'):
            #     self.build_share_memory()

            # with Tock('get data'):
            #     self.get_data_from_share_memory(cpu_count)
            self.data = self.data.reshape(-1)
            self.arr = to_ndarray(np.array_split(self.data, cpu_count))

            with Tock('mp'):
                processes = []
                for i in range(cpu_count):
                    p = mp.Process(target=self.job_1, args=(i, ))
                    p.start()
                    processes.append(p)
                for precess in processes:
                    precess.join()

    def mp_cost_pyarrow(self, cpu_count):
        with Tick('mp_cost_pyarrow'):
            #     with Tock('build share memory'):
            #         self.build_share_memory()

            #     with Tock('get data'):
            #         self.get_data_from_share_memory(cpu_count)
            self.data = self.data.reshape(-1)
            self.arr = to_ndarray(np.array_split(self.data, cpu_count))

            with Tock('mp'):
                processes = []
                for i in range(cpu_count):
                    p = mp.Process(target=self.job_2, args=(i, ))
                    p.start()
                    processes.append(p)
                for precess in processes:
                    precess.join()

    def plasma_memory(self, cpu_count):
        self.client = plasma.connect('/tmp/store')
        self.data = self.data.reshape(-1)
        object_id = self.client.put(to_ndarray(np.array_split(self.data, cpu_count)))
        return object_id

    def test_plasma_pyarrow(self, cpu_count):
        with Tick('test'):
            with Tock('connect'):
                object_id = self.plasma_memory(cpu_count)
            with Tock('plasma_memory'):
                tmp = self.client.get(object_id)
            with Tock('mp'):
                processes = []
                for i in range(cpu_count):
                    p = mp.Process(
                        target=self.job_3, args=(
                            tmp,
                            i,
                        )
                    )
                    p.start()
                    processes.append(p)
                for precess in processes:
                    precess.join()

    def test_plasma_pickle(self, cpu_count):
        with Tick('test'):
            with Tock('connect'):
                object_id = self.plasma_memory(cpu_count)
            with Tock('plasma_memory'):
                tmp = self.client.get(object_id)
            with Tock('mp'):
                processes = []
                #print(tmp)
                for i in range(cpu_count):
                    p = mp.Process(
                        target=self.job_4, args=(
                            tmp,
                            i,
                        )
                    )
                    p.start()
                    processes.append(p)
                for precess in processes:
                    precess.join()


if __name__ == '__main__':

    # data_1M =   np.random.random((3600,70)).astype(np.float32)
    # data_50M =  np.random.random((36000,350)).astype(np.float32)
    data_100M = np.random.random((36000, 700)).astype(np.float32)
    #data_150M = np.random.random((36000,700)).astype(np.float32)
    #data_200M = np.random.random((36000,1400)).astype(np.float32)
    #data_250M = np.random.random((36000,1750)).astype(np.float32)
    #data_300M = np.random.random((36000,2100)).astype(np.float32)
    #data_400M = np.random.random((36000,2800)).astype(np.float32)
    #data_500M = np.random.random((36000,3500)).astype(np.float32)
    #data_1G =   np.random.random((36000,7000)).astype(np.float32)
    #data_2G =   np.random.random((36000,14000)).astype(np.float32)
    #data_2dot5G = np.random.random((36000, 17500)).astype(np.float32)
    #data_3G =   np.random.random((36000,21000)).astype(np.float32)
    #data_4G =   np.random.random((72000,14000)).astype(np.float32)

    cpu_count = 6
    MT = Serialized_Zoo(data_100M)
    #MT.pickle_cost()
    #MT.pyarrow_cost()
    #MT.mp_cost_pickle(cpu_count)
    #MT.mp_cost_pyarrow(cpu_count)
    MT.test_plasma_pickle(cpu_count)
    #MT.test_plasma_pyarrow(cpu_count)
