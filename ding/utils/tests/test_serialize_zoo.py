from ding.utils.serialize_zoo import *
from ding.utils.import_helper import try_import_pyarrow,try_import_pickle



if __name__ == '__main__':

    #data_1M =   np.random.random((3600,70)).astype(np.float32)
    #data_50M =  np.random.random((36000,350)).astype(np.float32)
    #data_100M = np.random.random((36000, 700)).astype(np.float32)
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


    # random data  use pyarrow or pickle
    # pyarrow first ,if pyarrow uninstalled, use pickle
    # data = ['abc', {'1':np.array([1,2,3])}, 2, list(range(5))]
    # print(data)
    # A = SerializedZoo(data).dump()
    # B = SerializedZoo(A).load()
    # print(B)


    # all data type is same, data_100M = np.random.random((36000, 700)).astype(np.float32)  type = 'mp_pickle' or 'mp_pyarrow'
    data = np.random.random((36000, 700)).astype(np.float32)  #byte_length = 16800153
    shape = data.shape
    print(data)
    tmp, byte_length = SerializedZoo(data, shape, type='mp_pickle', cpu_count=6).dump()
    print(byte_length)
    result = SerializedZoo(tmp, shape, type='mp_pickle', cpu_count=6, byte_length = byte_length).load()
    result = to_ndarray(result).reshape(shape)
    print(result)


    # all data type is same and know bytes_shape, data_100M = np.random.random((36000, 700)).astype(np.float32)
    # Attention: pickle byte_length != pyarrow byte_length
    data = np.random.random((36000, 700)).astype(np.float32)
    shape = data.shape
    print(data)
    tmp, byte_length = SerializedZoo(data, shape, type='mp_pyarrow', cpu_count=6, byte_length=16800704).dump()
    result = SerializedZoo(tmp, shape, type='mp_pyarrow', cpu_count=6, byte_length = byte_length).load()
    result = to_ndarray(result).reshape(shape)
    print(result)







 
