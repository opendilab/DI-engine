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
from ding.utils.default_helper import yellow, cyan, Tick, Tock
from ding.utils.serialize_zoo import *

import pytest

# data example
# data_1M = np.random.random((3600, 70)).astype(np.float32)
# data_1M_tensor = torch.FloatTensor(data_1M)
# data_2M =  np.random.random((3600,140)).astype(np.float32)
# data_2M_tensor = torch.FloatTensor(data_2M)

#        |     nparray        |   tensor  |  tensor2nparray
#        |  pickle/pyarrow    |   pickle  |    pyarrow
#  2M   |  1.203ms/0.572ms   |   1.57ms  |   0.580(3e-3)ms
#  5M   |  106.97ms/11.99ms  |  143.37ms |   12.51(5e-3)ms
#  10M  |  218.10ms/23.59ms  |  297.67ms |   24.204(8e-3)ms


# In multiprocessing, build sharememory and put data in shm have large time cost  
# But in actual scenarios, we do not need build sharememory and put data in shm, just send data to other machines

#        |   pickle  |  cloudpickle  |  pyarrow  | mp_pickle | mp_pyarrow
#   1M   |   0.862ms |  1.285ms      |  0.534ms  | 116.88ms  | 155.35ms
#   2M   |   1.59ms  |  4.904ms      |  0.853ms  | 202.084ms | 294.94ms
#   5M   |   144.87ms|  230.3ms      |  51.25ms  | 4459.45ms | 6723.63ms



@pytest.mark.benchmark
def test_serialize_time():
    data = np.random.random((3600, 70)).astype(np.float32) # 1M
    #for func in ['test_mp_pickle']:
    for func in ['test_pickle','test_cloudpickle','test_pyarrow','test_mp_pickle','test_mp_pyarrow']:
        A = []
        B = []
        C = []
        print('----------warmup---------')
        for i in range(1):
            a,b,c = getattr(TestSerializeBenchmarkHelper, func)(data) 
        print('---------speed start------------')
        for i in range(1):
            a,b,c = getattr(TestSerializeBenchmarkHelper, func)(data)
            A.append(a.delta * 1000)
            B.append(b.delta * 1000)
            C.append(c.delta * 1000)
        print(sum(A) / len(A), sum(B) / len(B), sum(C) / len(C))
        print('---------speend end-------------')

@pytest.mark.benchmark
class TestSerializeBenchmarkHelper:

    def test_pickle(self):
        data=np.random.random((3600, 70)).astype(np.float32)  
        with Tick('pickle') as a:
            with Tock('ser') as b:
                A = SerializedZoo(data, type='pickle').dump()
            with Tock('des') as c:
                B = SerializedZoo(A, type='pickle').load()
        assert all(data[i][j] == B[i][j] for i in range(len(data)) for j in range(len(data[i])))
        return a,b,c
    
    def test_cloudpickle(self):
        data=np.random.random((3600, 70)).astype(np.float32)
        with Tick('cloudpickle') as a:
            with Tock('ser') as b:
                A = SerializedZoo(data, type='cloudpickle').dump()
            with Tock('des') as c:
                B = SerializedZoo(A, type='cloudpickle').load()
        assert all(data[i][j] == B[i][j] for i in range(len(data)) for j in range(len(data[i])))
        return a,b,c

    def test_pyarrow(self):
        data=np.random.random((3600, 70)).astype(np.float32)
        with Tick('pyarrow') as a:
            with Tock('ser') as b:
                A = SerializedZoo(data, type='pyarrow').dump()
            with Tock('des') as c:
                B = SerializedZoo(A, type='pyarrow').load()
        assert all(data[i][j] == B[i][j] for i in range(len(data)) for j in range(len(data[i])))
        return a,b,c

    def test_mp_pickle(self):
        data=np.random.random((3600, 70)).astype(np.float32)
        shape = data.shape
        with Tick('mp_pickle') as a:
            with Tock('ser') as b:
                A, byte_length = SerializedZoo(data, shape, type='mp_pickle', cpu_count=6).dump()
            with Tock('des') as c:
                B = SerializedZoo(A, shape, type='mp_pickle', cpu_count=6, byte_length=byte_length).load()
                B = to_ndarray(B).reshape(shape)
        assert all(data[i][j] == B[i][j] for i in range(len(data)) for j in range(len(data[i])))
        return a,b,c

    def test_mp_pyarrow(self):
        data=np.random.random((3600, 70)).astype(np.float32)
        shape = data.shape
        with Tick('mp_pyarrow') as a:
            with Tock('ser') as b:
                A, byte_length = SerializedZoo(data, shape, type='mp_pyarrow', cpu_count=6).dump()
            with Tock('des') as c:
                B = SerializedZoo(A, shape, type='mp_pyarrow', cpu_count=6, byte_length=byte_length).load()
                B = to_ndarray(B).reshape(shape).copy()
        assert all(data[i][j] == B[i][j] for i in range(len(data)) for j in range(len(data[i])))
        return a,b,c


