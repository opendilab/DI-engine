from s2clientprotocol import  sc2api_pb2 as sc_pb
import pkgutil

data_raw_3_16 = sc_pb.ResponseData()
data_3_16 = pkgutil.get_data('pysc2', 'lib/data/data_raw_3_16.serialized')
data_raw_3_16.ParseFromString(data_3_16)
#print(data_raw_3_16)

data_raw_4_0 = sc_pb.ResponseData()
data_4_0 = pkgutil.get_data('pysc2', 'lib/data/data_raw_4_0.serialized')
data_raw_4_0.ParseFromString(data_4_0)
#print(data_raw_4_0)

