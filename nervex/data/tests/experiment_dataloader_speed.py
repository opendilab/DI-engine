import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from functools import partial
from itertools import product

from nervex.data import AsyncDataLoader
from nervex.utils import EasyTimer

max_iter = 50

batch_size = 128
chunk_size = 32
num_workers = 4
use_cuda = False

process_time_args = [0.005, 0.03, 0.1]
read_infer_ratio_args = [1, 2, 4]
args = [item for item in product(*[process_time_args, read_infer_ratio_args])]

out_str_list = []


class MyDataset(Dataset):

    def __init__(self, process_time):
        self.data = torch.randn(256, 256)
        self.process_time = process_time

    def __len__(self):
        return batch_size * 300

    def __getitem__(self, idx):
        # todo read from file
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
        # x = self.main(x[0])  # todo real infer here?
        time.sleep(self.infer_time)
        return [x, idx]


def get_data_source(process_time):
    dataset = MyDataset(process_time)

    def data_source_fn(batch_size):
        return [partial(dataset.__getitem__, idx=i) for i in range(batch_size)]

    return data_source_fn


def entry(process_time, read_infer_ratio, use_cuda):
    out_str = '\n===== data: {:.4f}, model: {:.4f} ====='.format(process_time, process_time / read_infer_ratio)
    out_str_list.append(out_str)
    print(out_str)
    infer_time = process_time / read_infer_ratio
    model = MyModel(infer_time)
    if use_cuda:
        model.cuda()
    timer = EasyTimer()

    ##### Our DataLoader #####
    out_str = '----- Our DataLoader -----'
    out_str_list.append(out_str)
    print(out_str)
    data_source = get_data_source(process_time)
    device = 'cuda' if use_cuda else 'cpu'
    our_dataloader = AsyncDataLoader(data_source, batch_size, device, num_workers=num_workers, chunk_size=chunk_size)
    iter = 0
    total_data_time = 0.
    start_time = time.time()
    while True:
        with timer:
            data = next(our_dataloader)
        data_time = timer.value
        if iter > 5:  # ignore start-5-iter time
            total_data_time += data_time
        with timer:
            with torch.no_grad():
                _, idx = model(data)
        model_time = timer.value
        total_time = data_time + model_time
        print('iter {:0>2d}, total_time: {:.4f}, data_time: {:.4f}, model_time: {:.4f}'.format(iter, total_time, data_time, model_time))
        iter += 1
        if iter == max_iter:
            break
    total_time = time.time() - start_time
    out_str = 'total_time: {:.4f}, total_data_time: {:.4f}'.format(total_time, total_data_time)
    out_str_list.append(out_str)
    print(out_str)
    our_dataloader.__del__()

    ##### PyTorch DataLoader #####
    out_str = '----- PyTorch DataLoader -----'
    out_str_list.append(out_str)
    print(out_str)
    dataset = MyDataset(process_time)
    torch_dataloader = DataLoader(dataset, batch_size, num_workers=num_workers)
    torch_dataloader_iter = torch_dataloader.__iter__()
    data = next(torch_dataloader_iter)
    iter = 0
    total_data_time = 0.
    start_time = time.time()
    while True:
        with timer:
            data = next(torch_dataloader_iter)
            if use_cuda:
                data = data.cuda()
        data_time = timer.value
        if iter > 5:  # ignore start-5-iter time
            total_data_time += data_time
        with timer:
            with torch.no_grad():
                _, idx = model(data)
        model_time = timer.value
        total_time = data_time + model_time
        print('iter {:0>2d}, total_time: {:.4f}, data_time: {:.4f}, model_time: {:.4f}'.format(iter, total_time, data_time, model_time))
        iter += 1
        if iter == max_iter:
            break
    total_time = time.time() - start_time
    out_str = 'total_time: {:.4f}, total_data_time: {:.4f}'.format(total_time, total_data_time)
    out_str_list.append(out_str)
    print(out_str)


if __name__ == "__main__":
    for process_time, read_infer_ratio in args:
        entry(process_time, read_infer_ratio, use_cuda=use_cuda)
        torch.cuda.empty_cache()
    print("\n".join(out_str_list))
