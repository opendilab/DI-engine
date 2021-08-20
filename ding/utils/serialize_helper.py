

from multiprocessing.dummy import Value
import numpy as np
import pickle
import multiprocessing as mp
from ding.torch_utils.data_helper import to_list, to_ndarray
from multiprocessing import Process, Manager,Pipe, connection, get_context, Array, cpu_count,Lock, RawArray
import ctypes
from utils import *
import pyarrow
import pyarrow.plasma as plasma
import mmap
import contextlib


def bold(x):       return '\033[1m'  + str(x) + '\033[0m'
def dim(x):        return '\033[2m'  + str(x) + '\033[0m'
def italicized(x): return '\033[3m'  + str(x) + '\033[0m'
def underline(x):  return '\033[4m'  + str(x) + '\033[0m'
def blink(x):      return '\033[5m'  + str(x) + '\033[0m'
def inverse(x):    return '\033[7m'  + str(x) + '\033[0m'
def gray(x):       return '\033[90m' + str(x) + '\033[0m'
def red(x):        return '\033[91m' + str(x) + '\033[0m'
def green(x):      return '\033[92m' + str(x) + '\033[0m'
def yellow(x):     return '\033[93m' + str(x) + '\033[0m'
def blue(x):       return '\033[94m' + str(x) + '\033[0m'
def magenta(x):    return '\033[95m' + str(x) + '\033[0m'
def cyan(x):       return '\033[96m' + str(x) + '\033[0m'
def white(x):      return '\033[97m' + str(x) + '\033[0m'


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
        self.delta = self.t_end-self.t_start
        self.fps = 1/self.delta

        if not self.silent:
            print(yellow('[%.3fs]' % (self.delta), ), flush=True)

class Tock():
    def __init__(self, name=None, report_time=True):
        self.name = '' if name == None else name+':'
        self.report_time = report_time

    def __enter__(self):
        self.t_start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.t_end = time.time()
        self.delta = self.t_end-self.t_start
        self.fps = 1/self.delta
        if self.report_time:
            print(yellow(self.name)+cyan('%.3fs'%(self.delta)), end=' ', flush=True)
        else:
            print(yellow('.'), end='', flush=True)



class MultiprocessingPickle:

    def __init__(self, data, cpu_count, mode='dump') -> None:
        self.data = data
        self.cpu_count = cpu_count
        self.mode = mode
        self.dump = []
        self.load = []
        if self.mode == 'dump':
            self.pack()
        if self.mode == 'load':
            self.unpack()

    def job_serialized(self,i,buffer):


        with buffer.get_lock():
            print(i)
            print(self.arr[i])
            serialized_x = pickle.dumps(self.arr[i], protocol=pickle.HIGHEST_PROTOCOL)
            print(serialized_x)
            print(type(serialized_x))
            buffer[i]=serialized_x
    
        if i==0:
            buffer[i]=b'1234'
        if i==1:
            buffer[i]=serialized_x
        if i==2:
            buffer[i]=b'ty'
        if i==3:
            buffer[i]=b'567789'
        if i==4:
            buffer[0]=b'bbbbbbvvvvvv'

        #     buffer[i*16800153:(i+1)*16800153] = serialized_x[:]
        #     print(buffer[i*16800153:(i+1)*16800153])
            # for j in range(i*16800153,(i+1)*16800153):
            #     #print(j)
            #     #arr[j] = serialized_x[j-i*16800153]
            #     buffer[j] = serialized_x[j-i*16800153]
            # print(type(serialized_x))
            # print(len(serialized_x))
            #print(arr[i*16800153:(i+1)*16800153])
        
            
            # print(dir(buffer))
            # print(len(buffer))
            # print('-----')
    
    def job_deserialized(self,i,return_list):
        deserialized_x = pickle.loads(self.data[i])
        return_list=deserialized_x
    
    def pack(self):
        self.data =self.data.reshape(-1)
        self.arr = to_ndarray(np.array_split(self.data,self.cpu_count))
        processes = []

        buffer = Array(ctypes.c_char_p,6)
        
        for i in range(self.cpu_count):
            p = mp.Process(target=self.job_serialized,args=(i,buffer))
            p.start()
            processes.append(p)
        for precess in processes:
            precess.join()
        print('---')
        print(buffer[:])
        exit()
    
    def unpack(self):
        processes = []
        buffer = mp.Array(ctypes.c_double, 252000000)
        return_arr = np.frombuffer(buffer.get_obj(), dtype=ctypes.c_double).reshape((6,42000000))

        pool = mp.Pool(processes=self.cpu_count)
        for i in range(self.cpu_count):
            p = mp.Process(target=self.job_deserialized,args=(i,return_arr,))
            p.start()
            processes.append(p)
        for precess in processes:
            precess.join()
        self.load = return_arr

class MultiprocessingPyarrow:

    def __init__(self, data, cpu_count, mode='dump') -> None:
        self.data = data
        self.cpu_count = cpu_count
        self.mode = mode
        self.dump = []
        self.load = []
        if self.mode == 'dump':
            self.pack()
        if self.mode == 'load':
            self.unpack()

    def job_serialized(self,i,return_list):
        serialized_x = pyarrow.serialize(self.arr[i]).to_buffer()
        return_list[i]=serialized_x
    
    def job_deserialized(self,i,return_list):
        deserialized_x = pyarrow.deserialize(self.data[i])
        return_list[i]=deserialized_x
    
    def pack(self):
        self.data =self.data.reshape(-1)
        self.arr = to_ndarray(np.array_split(self.data,self.cpu_count))
        processes = []
        manager = Manager()
        return_list = manager.list([0]*self.cpu_count)
        pool = mp.Pool(processes=self.cpu_count)
        for i in range(self.cpu_count):
            p = mp.Process(target=self.job_serialized,args=(i,return_list,))
            p.start()
            processes.append(p)
        for precess in processes:
            precess.join()
        self.dump = return_list
        print(self.dump)
    
    def unpack(self):
        processes = []
        manager = Manager()
        return_list = manager.list([0]*self.cpu_count)
        pool = mp.Pool(processes=self.cpu_count)
        for i in range(self.cpu_count):
            p = mp.Process(target=self.job_deserialized,args=(i,return_list,))
            p.start()
            processes.append(p)
        for precess in processes:
            precess.join()
        self.load = return_list