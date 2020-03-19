import os
import sys
import ceph
import torch
from io import BytesIO

# p = 'Protoss_Zerg_3502_0bcbee57762866600654d160fa8698a9af3ddd0fb80d60dcc9b95ae83a71ec7e.meta'
p = 'Zerg_Terran_3541_68f42a612b92c4520341ad375bec6a7ee4981a26740664177324a5d61b3db43a.meta'

prefix1 = '/mnt/lustre/zhangming/data/Replays_decode_valid/'
prefix2 = 's3://mybucket/'

name1 = prefix1 + p
name2 = prefix2 + p

# data1 = torch.load(name1)
# print(data1)
# print('--------------------------------------')
# print('--------------------------------------')
# print('--------------------------------------')

s3client = ceph.S3Client()

s3client = ceph.S3Client(access_key="GWXHDG7SGDAW34Q71R2F", secret_key="v8fdUvycLIvW64ok30MBcXCtJ49vFmbgIIVNaMU2")

data2 = s3client.Get(name2)

data2 = torch.load(BytesIO(data2))

print(data2)
