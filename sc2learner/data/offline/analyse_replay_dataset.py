import os
import sys
import torch
import numpy as np
import yaml
from easydict import EasyDict
import random
from torch.utils.data import DataLoader
from collections import deque
import time
import threading
import multiprocessing

from replay_dataset import ReplayDataset, policy_collate_fn
from sc2learner.utils import read_file_ceph


def get_dataset_config():
    with open('/mnt/lustre/zhangming/workspace/software/SenseStar-refactoring/' +
              'sc2learner/worker/learner/alphastar_sl_learner_default_config.yaml', "r") as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)
    config.data.train.replay_list = '/mnt/lustre/zhangming/data/train/Zerg_None_None_3500_train_5200.txt.5.local'
    return config.data.train


def build(dataset_config, train_dataset=True):
    dataset = ReplayDataset(dataset_config, train_mode=train_dataset)
    sampler = None
    shuffle = False
    # set num_workers=0 for preventing ceph reading file bug
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        pin_memory=False,
        num_workers=0,
        sampler=sampler,
        shuffle=shuffle,
        drop_last=False,
        collate_fn=policy_collate_fn
    )
    return dataloader


def analyse_replay_dataset():
    dataset_config = get_dataset_config()
    dataset_config.use_ceph = False
    replay_dataloader = build(dataset_config, train_dataset=True)

    replay_dataloader.dataset.step()
    for idx, batch_data in enumerate(replay_dataloader):
        print(idx, type(batch_data), sys.getsizeof(batch_data))


def test_serial():
    p = "/mnt/lustre/zhangming/data/Zerg_None_None_3500_train_5200.txt"
    sum = 0
    lines = open(p, 'r').readlines()
    for index, line in enumerate(lines[:20]):
        t1 = time.time()
        data = read_file_ceph(line.strip() + ".step")
        t2 = time.time()
        sum += t2 - t1
        print("{} cost {}".format(index, t2 - t1))
    print("[serial] total time {}, avg time = {}".format(sum, sum / 20))


parallel_sum = 0


def process_thread(index, file_path):
    t1 = time.time()
    data = read_file_ceph(file_path.strip() + ".step")
    t2 = time.time()
    print("{} cost {}".format(index, t2 - t1))
    return t2 - t1


def process_callback(t):
    global parallel_sum
    parallel_sum += t


def test_parallel_thread():
    p = "/mnt/lustre/zhangming/data/Zerg_None_None_3500_train_5200.txt"
    sum = 0
    thread_list = []
    lines = open(p, 'r').readlines()
    for index, line in enumerate(lines[:20]):
        thread_list.append(threading.Thread(target=process_thread, args=(index, line.strip())))

    t1 = time.time()
    for t in thread_list:
        t.start()
    for t in thread_list:
        t.join()
    t2 = time.time()
    print("totally cost {}".format(t2 - t1))


def test_parallel_process():
    global parallel_sum
    p = "/mnt/lustre/zhangming/data/Zerg_None_None_3500_train_5200.txt"
    lines = open(p, 'r').readlines()
    p = multiprocessing.Pool(12)
    t1 = time.time()
    for index, line in enumerate(lines[:20]):
        r = p.apply_async(process_thread, args=(index, line.strip()), callback=process_callback)
    p.close()
    p.join()
    t2 = time.time()
    print("[parallel] totally time {}, avg time {}".format(t2 - t1, (t2 - t1) / 20))


if __name__ == '__main__':
    # analyse_replay_dataset()

    # test_parallel_process()
    # print("-----------------------------------")
    # test_serial()

    test_serial()
    print("-----------------------------------")
    test_parallel_process()
