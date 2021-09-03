from ding.utils.serialize_zoo import SerializedZoo
from ding.utils.import_helper import try_import_pyarrow, try_import_pickle
import pytest
from ding.torch_utils.data_helper import to_list, to_ndarray
import numpy as np


@pytest.mark.unittest
class TestSerializeHelper:

    # 1. mixed data use pyarrow/pickle, if pyarrow uninstalled, use pickle
    # 2. data type is fixed but byte_length is not known, for example test_mp_pickle
    # 3. data type is fixed but byte_length is known, for example test_mp_pyarrow
    # Attention: pickle byte_length != pyarrow byte_length

    def test_pickle(self):
        data = np.random.random((3600, 700)).astype(np.float32)
        A = SerializedZoo(data, type='pickle').dump()
        B = SerializedZoo(A, type='pickle').load()
        assert all(data[i][j] == B[i][j] for i in range(len(data)) for j in range(len(data[i])))

    def test_pyarrow(self):
        data = np.random.random((3600, 700)).astype(np.float32)
        A = SerializedZoo(data, type='pyarrow').dump()
        B = SerializedZoo(A, type='pyarrow').load()
        assert all(data[i][j] == B[i][j] for i in range(len(data)) for j in range(len(data[i])))

    def test_mp_pickle(self):
        data = np.random.random((3600, 700)).astype(np.float32)
        shape = data.shape
        A, byte_length = SerializedZoo(data, shape, type='mp_pickle', cpu_count=6).dump()
        B = SerializedZoo(A, shape, type='mp_pickle', cpu_count=6, byte_length=byte_length).load()
        B = to_ndarray(B).reshape(shape)
        assert all(data[i][j] == B[i][j] for i in range(len(data)) for j in range(len(data[i])))

    def test_mp_pyarrow(self):
        data = np.random.random((3600, 700)).astype(np.float32)
        shape = data.shape
        A, byte_length = SerializedZoo(data, shape, type='mp_pyarrow', cpu_count=6, byte_length=1680704).dump()
        B = SerializedZoo(A, shape, type='mp_pyarrow', cpu_count=6, byte_length=byte_length).load()
        B = to_ndarray(B).reshape(shape).copy()
        assert all(data[i][j] == B[i][j] for i in range(len(data)) for j in range(len(data[i])))
