import os
import sys
import torch
import copy
import pickle
import zlib
import multiprocessing
import time
import traceback

from sc2learner.envs import compress_obs, decompress_obs
from sc2learner.utils import read_file_ceph, save_file_ceph

import zlib


def zlib_compressor(obs):
    return zlib.compress(pickle.dumps(obs))


def zlib_decompressor(local_saev_path):
    data = torch.load(local_save_path)
    new_data = []
    for item in data:
        new_data.append(pickle.loads(zlib.decompress(item)))
    return new_data


def replay_fix(ceph_root, replay_path, ceph_save_root, local_save_path):
    def fix(data, n):
        data = data.reshape(*data.shape[1:], data.shape[0])
        data = data.permute(2, 0, 1).contiguous()
        return data

    t_zlib = 0

    t1 = time.time()
    d1 = read_file_ceph(ceph_root + replay_path)
    t2 = time.time()
    replay = torch.load(d1)
    t3 = time.time()
    dim_list = [2, 4, 2, 5, 2, 2, 2]
    new_data = []
    new_data_zlib = []
    for idx, d in enumerate(replay):
        new_d = copy.deepcopy(decompress_obs(d))
        idx = 1
        for t in dim_list:
            new_d['spatial_info'][idx:idx + t] = fix(new_d['spatial_info'][idx:idx + t], t)
            idx = idx + t
        compress_item = compress_obs(new_d)
        new_data.append(compress_item)
        t_s = time.time()
        new_data_zlib.append(zlib_compressor(compress_item))
        t_zlib += time.time() - t_s

    # output path should be modified
    t4 = time.time()
    save_file_ceph(ceph_save_root, replay_path, new_data)
    t5 = time.time()
    # new_data_load = read_file_ceph(ceph_save_root + replay_path, read_type='pickle')
    # print(len(new_data), len(new_data_load))
    torch.save(new_data_zlib, local_save_path)
    t6 = time.time()
    # new_data_zlib_load = torch.load(local_save_path)
    # print(new_data_zlib)
    # print("----new_data")
    # print(new_data[0])
    # print("----new_data_load")
    # print(new_data_load[0])
    # print("----new_data_zlib")
    # print(new_data_zlib[0])
    # print("----new_data_zlib_decompress")
    # print(pickle.loads(zlib.decompress(new_data_zlib_load[0])))
    print(
        (
            "{} totally {:.3f} download {:.3f}, torch.load {:.3f}, " +
            "process {:.3f} (zlib {:.3f}), upload {:.3f}, torch.save {:.3f}"
        ).format(replay_path, t6 - t1, t2 - t1, t3 - t2, t4 - t3, t_zlib, t5 - t4, t6 - t5)
    )


if __name__ == '__main__':
    replay_list = sys.argv[1]
    ceph_root = 's3://replay_decode_410_clean/'
    ceph_save_root = 's3://replay_decode_410_clean_2/'
    # replay_list = '/mnt/lustre/zhangming/data/listtempfix/replay_decode_410_clean.list'
    local_save_root = '/mnt/lustre/zhangming/data/replay_decode_410_clean_compress'

    if not os.path.isdir(local_save_root):
        os.mkdir(local_save_root)

    print(replay_list)

    # p = multiprocessing.Pool(2)

    lines = open(replay_list, 'r').readlines()
    for index, line in enumerate(lines):
        # if index % 10 == 0:
        replay_path = line.strip()
        local_save_path = os.path.join(local_save_root, replay_path)
        try:
            replay_fix(ceph_root, replay_path, ceph_save_root, local_save_path)
        except Exception as e:
            print(''.join(traceback.format_tb(e.__traceback__)))
            print("[error] {}".format(sys.exc_info()))
    #     r = p.apply_async(replay_fix, args=(ceph_root, replay_path, ceph_save_root, local_save_path))
    # p.close()
    # p.join()
