import numpy as np
from ding.torch_utils.data_helper import to_list, to_ndarray
from ding.utils.serialize_helper import MultiprocessingPickle, MultiprocessingPyarrow, Tick, Tock
from ding.utils.import_helper import try_import_pyarrow,try_import_pickle
import cloudpickle


pyarrow = try_import_pyarrow()
if not pyarrow:
    pickle = try_import_pickle()

class SerializedZoo:
    def __init__(self, var, shape=(0,), type='pyarrow', cpu_count=1, byte_length = -1) -> None:
        self.var = var
        self.type = type
        self.cpu_count = cpu_count
        self.shape = shape
        self.byte_length =byte_length

    def dump(self):
        self.var = to_ndarray(self.var)
        if self.type == 'cloudpickle':
            return cloudpickle.dumps(self.var)

        if self.type == 'pyarrow':
            if pyarrow:
                return pyarrow.serialize(self.var).to_buffer()
            else:
                return pickle.dumps(self.var, protocol=pickle.HIGHEST_PROTOCOL)

        if self.type == 'mp_pickle':
            return MultiprocessingPickle(self.var, self.shape, self.cpu_count,mode='dump',byte_length = self.byte_length).pack()

        if self.type == 'mp_pyarrow':
            return MultiprocessingPyarrow(self.var, self.shape, self.cpu_count,mode='dump',byte_length = self.byte_length).pack()


    def load(self):
        if self.type == 'cloudpickle':
            return cloudpickle.loads(self.var)

        if self.type == 'pyarrow':
            if pyarrow:
                return pyarrow.deserialize(self.var)
            else:
                return pickle.loads(self.var)

        if self.type == 'mp_pickle':
            return MultiprocessingPickle(self.var, self.shape, self.cpu_count, mode='load',byte_length = self.byte_length).unpack()

        if self.type == 'mp_pyarrow':
            return MultiprocessingPyarrow(self.var, self.shape, self.cpu_count, mode='load',byte_length = self.byte_length).unpack()