from EfficiencyOptimization.utils import Tock
import multiprocessing
import cloudpickle
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union
import numpy as np
import pickle
from ding.torch_utils.data_helper import to_list, to_ndarray
import pyarrow

from serialize_helper import MultiprocessingPickle, MultiprocessingPyarrow, Tick, Tock
#MultipcrocessingPickleWithPlasma, MultipcrocessingPyarrowWithPlasma




class SerializedZoo:
    def __init__(self, var, type='pickle', cpu_count=1) -> None:
        self.var = var
        self.type = type
        self.cpu_count = cpu_count
    
    def dump(self): 
        self.var = to_ndarray(self.var)
        if self.type == 'pickle':
            return pickle.dumps(self.var, protocol=pickle.HIGHEST_PROTOCOL)

        if self.type == 'cloudpickle':
            return cloudpickle.dumps(self.var)

        if self.type == 'pyarrow':
            return pyarrow.serialize(self.var).to_buffer()

        if self.type == 'mp_pickle':
            return MultiprocessingPickle(self.var,self.cpu_count).dump
        
        if self.type == 'mp_pyarrow':
            return MultiprocessingPyarrow(self.var,self.cpu_count).dump
        
        if self.type =='mp_pickle_plasma':
            return MultiprocessingPickleWithPlasma(self.var,self.cpu_count)
        
        if self.type =='mp_pyarrow_plasma':
            return MultiprocessingPyarrowWithPlasma(self.var,self.cpu_count)
    
    def load(self):
        if self.type == 'pickle':
            return pickle.loads(self.var)

        if self.type == 'cloudpickle':
            return cloudpickle.loads(self.var)

        if self.type == 'pyarrow':
            return pyarrow.deserialize(self.var)

        if self.type == 'mp_pickle':
            return MultiprocessingPickle(self.var,self.cpu_count,mode = 'load').load
        
        if self.type == 'mp_pyarrow':
            return MultiprocessingPyarrow(self.var,self.cpu_count,mode = 'load').load
        
        if self.type =='mp_pickle_plasma':
            return MultipcrocessingPickleWithPlasma(self.var,self.cpu_count,mode = 'load')
        
        if self.type =='mp_pyarrow_plasma':
            return MultipcrocessingPyarrowWithPlasma(self.var,self.cpu_count,mode = 'load')



if __name__ == '__main__':

    # data_1M =   np.random.random((3600,70)).astype(np.float32)  
    # data_50M =  np.random.random((36000,350)).astype(np.float32) 
    data_100M = np.random.random((36000,700)).astype(np.float32) 
    #data_150M = np.random.random((36000,700)).astype(np.float32) 
    #data_200M = np.random.random((36000,1400)).astype(np.float32) 
    #data_250M = np.random.random((36000,1750)).astype(np.float32) 
    #data_300M = np.random.random((36000,2100)).astype(np.float32) 
    #data_400M = np.random.random((36000,2800)).astype(np.float32) 
    #data_500M = np.random.random((36000,3500)).astype(np.float32) 
    #data_1G =   np.random.random((36000,7000)).astype(np.float32) 
    #data_2G =   np.random.random((36000,14000)).astype(np.float32) 
    #data_2dot5G = np.random.random((36000,17500)).astype(np.float32) 
    #data_3G =   np.random.random((36000,21000)).astype(np.float32) 
    #data_4G =   np.random.random((72000,14000)).astype(np.float32) 


    #data = data_100M
    data = np.random.randint(10,size=(6,1)).astype(np.float32) 

    with Tick('pickle'):
        with Tock('dump'): 
            A = SerializedZoo(data,type ='pickle').dump()
        with Tock('load'):
            B = SerializedZoo(A,type ='pickle').load()


    # with Tick('cloudpickle'):
    #     with Tock('dump'): 
    #         A = SerializedZoo(data,type ='cloudpickle',cpu_count=6).dump()
    #     with Tock('load'):
    #         B = SerializedZoo(A,type ='cloudpickle',cpu_count=6).load()


    # with Tick('pyarrow'):
    #     with Tock('dump'): 
    #         A = SerializedZoo(data,type ='pyarrow',cpu_count=6).dump()
    #     with Tock('load'):
    #         B = SerializedZoo(A,type ='pyarrow',cpu_count=6).load()


    with Tick('mp_pickle'):
        with Tock('dump'): 
            A = SerializedZoo(data,type ='mp_pickle',cpu_count=6).dump()
            #print(A)
        with Tock('load'):
            B = SerializedZoo(A,type ='mp_pickle',cpu_count=6).load()


    # with Tick('mp_pyarrow'):
    #     with Tock('dump'): 
    #         A = SerializedZoo(data,type ='mp_pyarrow',cpu_count=6).dump()
    #     with Tock('load'):
    #         B = SerializedZoo(A,type ='mp_pyarrow',cpu_count=6).load()