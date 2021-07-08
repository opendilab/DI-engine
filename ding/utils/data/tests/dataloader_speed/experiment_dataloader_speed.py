import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from functools import partial
from itertools import product
import os.path as osp
import os
import random

from ding.utils import EasyTimer, read_file
from ding.utils.data import AsyncDataLoader

exp_times = 10
max_iter = 50
num_workers = 8
use_cuda = True

# read_file_time, process_time, batch_size, chunk_size, env_name
env_args = [
    (0.0008, 0.005, 128, 32, "small"),
    (0.0008, 0.05, 64, 16, "middle"),
    (0.6, 0.2, 4, 1, "big16"),
    (2, 0.25, 4, 1, "big64"),
]
data_infer_ratio_args = [1, 2, 4]

args = [item for item in product(*[env_args, data_infer_ratio_args])]

out_str_list = []


class MyDataset(Dataset):

    def __init__(self, file_time, process_time, batch_size, name):
        self.data = torch.randn(256, 256)
        self.file_time = file_time
        self.process_time = process_time
        self.batch_size = batch_size
        self.path = osp.join(osp.dirname(__file__), "../traj_files/{}/data".format(name))
        self.file_list = os.listdir(self.path)
        self.file_sequence = random.sample(range(0, len(self.file_list)), len(self.file_list))
        self.i = 0

    def __len__(self):
        return self.batch_size * max_iter * 2

    def __getitem__(self, idx):
        try:
            s = read_file(osp.join(self.path, self.file_list[self.file_sequence[self.i]]))
        except:
            print("file read meets an error")
            time.sleep(self.file_time)
        self.i = (self.i + 1) % len(self.file_list)
        time.sleep(self.process_time)
        return [self.data, idx]


class MyModel(nn.Module):

    def __init__(self, infer_time):
        super().__init__()
        self.main = [nn.Linear(256, 256) for _ in range(10)]
        self.main = nn.Sequential(*self.main)
        self.infer_time = infer_time

    def forward(self, x):
        idx = x[1]
        # No real infer here.
        time.sleep(self.infer_time)
        return [x, idx]


def get_data_source(dataset):

    def data_source_fn(batch_size):
        return [partial(dataset.__getitem__, idx=i) for i in range(batch_size)]

    return data_source_fn


def entry(env, read_infer_ratio, use_cuda):
    file_time, process_time, batch_size, chunk_size, data_name = env[0], env[1], env[2], env[3], env[4]
    data_time = file_time + process_time
    infer_time = data_time * (batch_size / num_workers) * 1.05 / read_infer_ratio
    out_str = '\n===== each_data: {:.4f}({}), infer: {:.4f}, read/infer: {:.4f}, \
        batch_size: {}, chunk_size: {} ====='.format(
        data_time, data_name, infer_time, read_infer_ratio, batch_size, chunk_size
    )
    out_str_list.append(out_str)
    print(out_str)

    model = MyModel(infer_time)
    if use_cuda:
        model.cuda()
    timer = EasyTimer()

    # ### Our DataLoader ####
    total_sum_time_list = []
    total_data_time_list = []
    total_infer_time_list = []
    for _ in range(exp_times):
        print('\t----- Our DataLoader -----')
        dataset = MyDataset(file_time, process_time, batch_size, data_name)
        data_source = get_data_source(dataset)
        device = 'cuda' if use_cuda else 'cpu'
        our_dataloader = AsyncDataLoader(
            data_source, batch_size, device, num_workers=num_workers, chunk_size=chunk_size
        )
        iter = 0
        total_data_time = 0.
        total_infer_time = 0.
        total_sum_time = 0.
        while True:
            with timer:
                data = next(our_dataloader)
            data_time = timer.value
            with timer:
                with torch.no_grad():
                    _, idx = model(data)
            infer_time = timer.value
            sum_time = data_time + infer_time
            if iter > 5:  # ignore start-5-iter time
                total_data_time += data_time
                total_infer_time += infer_time
            print(
                '\t\titer {:0>2d}, sum_time: {:.4f}, data_time: {:.4f}, infer_time: {:.4f}'.format(
                    iter, sum_time, data_time, infer_time
                )
            )
            iter += 1
            if iter == max_iter:
                break
        total_sum_time = total_data_time + total_infer_time
        out_str = '\ttotal_sum_time: {:.4f}, total_data_time: {:.4f}, \
            total_infer_time: {:.4f}, data/sum: {:.4f}'.format(
            total_sum_time, total_data_time, total_infer_time, total_data_time / total_sum_time
        )
        # out_str_list.append(out_str)
        print(out_str)
        our_dataloader.__del__()
        torch.cuda.empty_cache()

        total_sum_time_list.append(total_sum_time)
        total_data_time_list.append(total_data_time)
        total_infer_time_list.append(total_infer_time)
    total_sum_time = sum(total_sum_time_list) / len(total_sum_time_list)
    total_data_time = sum(total_data_time_list) / len(total_data_time_list)
    total_infer_time = sum(total_infer_time_list) / len(total_infer_time_list)
    out_str = '\t(Our DataLoader {} average) total_sum_time: {:.4f}, \
        total_data_time: {:.4f}, total_infer_time: {:.4f}, data/sum: {:.4f}'.format(
        exp_times, total_sum_time, total_data_time, total_infer_time, total_data_time / total_sum_time
    )
    out_str_list.append(out_str)
    print(out_str)

    # ### PyTorch DataLoader ####
    for real_num_workers in [0, 8]:
        total_sum_time_list = []
        total_data_time_list = []
        total_infer_time_list = []
        for _ in range(exp_times):
            print('\t----- PyTorch DataLoader (num_workers = {}) -----'.format(real_num_workers))
            dataset = MyDataset(file_time, process_time, batch_size, data_name)
            torch_dataloader = DataLoader(dataset, batch_size, num_workers=real_num_workers)
            torch_dataloader_iter = torch_dataloader.__iter__()
            iter = 0
            total_data_time = 0.
            total_infer_time = 0.
            total_sum_time = 0.
            while True:
                with timer:
                    data = next(torch_dataloader_iter)[0]
                    if use_cuda:
                        data = data.cuda()
                data_time = timer.value
                with timer:
                    with torch.no_grad():
                        _, idx = model(data)
                infer_time = timer.value
                sum_time = data_time + infer_time
                if iter > 5:  # ignore start-5-iter time
                    total_data_time += data_time
                    total_infer_time += infer_time
                print(
                    '\t\titer {:0>2d}, sum_time: {:.4f}, data_time: {:.4f}, infer_time: {:.4f}'.format(
                        iter, sum_time, data_time, infer_time
                    )
                )
                iter += 1
                if iter == max_iter:
                    break
            total_sum_time = total_data_time + total_infer_time
            out_str = '\ttotal_sum_time: {:.4f}, total_data_time: {:.4f}, \
                total_infer_time: {:.4f}, data/sum: {:.4f}'.format(
                total_sum_time, total_data_time, total_infer_time, total_data_time / total_sum_time
            )
            # out_str_list.append(out_str)
            print(out_str)
            torch.cuda.empty_cache()

            total_sum_time_list.append(total_sum_time)
            total_data_time_list.append(total_data_time)
            total_infer_time_list.append(total_infer_time)
        total_sum_time = sum(total_sum_time_list) / len(total_sum_time_list)
        total_data_time = sum(total_data_time_list) / len(total_data_time_list)
        total_infer_time = sum(total_infer_time_list) / len(total_infer_time_list)
        out_str = '\t(PyTorch DataLoader baseline {} average) total_sum_time: {:.4f}, \
            total_data_time: {:.4f}, total_infer_time: {:.4f}, data/sum: {:.4f}'.format(
            exp_times, total_sum_time, total_data_time, total_infer_time, total_data_time / total_sum_time
        )
        out_str_list.append(out_str)
        print(out_str)


if __name__ == "__main__":
    for env, read_infer_ratio in args:
        entry(env, read_infer_ratio, use_cuda=use_cuda)
    print("\n".join(out_str_list))
