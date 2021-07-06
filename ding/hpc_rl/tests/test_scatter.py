import time
import torch
from typing import Tuple
from hpc_rll.origin.scatter_connection import ScatterConnection
from hpc_rll.torch_utils.network.scatter_connection import ScatterConnection as HPCScatterConnection
from testbase import mean_relative_error, times

assert torch.cuda.is_available()
use_cuda = True

B = 256
M = 256
N = 256
H = 16
W = 16


# Note: origin gpu version of cover mode is not determinate, thus validation test use origin cpu version instead
def scatter_val():
    for scatter_type in ['add', 'cover']:
        ori_input = torch.randn(B, M, N)
        h = torch.randint(
            low=0, high=H, size=(
                B,
                M,
            )
        ).unsqueeze(dim=2)
        w = torch.randint(
            low=0, high=W, size=(
                B,
                M,
            )
        ).unsqueeze(dim=2)
        ori_location = torch.cat([h, w], dim=2)
        ori_scatter = ScatterConnection(scatter_type)

        hpc_input = ori_input.clone().detach()
        hpc_location = ori_location.clone().detach()
        hpc_scatter = HPCScatterConnection(B, M, N, H, W, scatter_type)

        if use_cuda:
            #ori_input = ori_input.cuda()
            #ori_location = ori_location.cuda()
            #ori_scatter = ori_scatter.cuda()

            hpc_input = hpc_input.cuda()
            hpc_location = hpc_location.cuda()
            hpc_scatter = hpc_scatter.cuda()

        ori_input.requires_grad_(True)
        ori_output = ori_scatter(ori_input, (H, W), ori_location)
        ori_loss = ori_output * ori_output
        ori_loss = ori_loss.mean()
        ori_loss.backward()
        if use_cuda:
            torch.cuda.synchronize()

        hpc_input.requires_grad_(True)
        hpc_output = hpc_scatter(hpc_input, hpc_location)
        hpc_loss = hpc_output * hpc_output
        hpc_loss = hpc_loss.mean()
        hpc_loss.backward()
        if use_cuda:
            torch.cuda.synchronize()

        mre = mean_relative_error(
            torch.flatten(ori_loss).cpu().detach().numpy(),
            torch.flatten(hpc_loss).cpu().detach().numpy()
        )
        print("scatter type {} fp mean_relative_error: {}".format(scatter_type, str(mre)))
        mre = mean_relative_error(
            torch.flatten(ori_input.grad).cpu().detach().numpy(),
            torch.flatten(hpc_input.grad).cpu().detach().numpy()
        )
        print("scatter type {} bp mean_relative_error: {}".format(scatter_type, str(mre)))


# Note: performance test use origin gpu version
def scatter_perf():
    for scatter_type in ['add', 'cover']:
        ori_input = torch.randn(B, M, N)
        h = torch.randint(
            low=0, high=H, size=(
                B,
                M,
            )
        ).unsqueeze(dim=2)
        w = torch.randint(
            low=0, high=W, size=(
                B,
                M,
            )
        ).unsqueeze(dim=2)
        ori_location = torch.cat([h, w], dim=2)
        ori_scatter = ScatterConnection(scatter_type)

        hpc_input = ori_input.clone().detach()
        hpc_location = ori_location.clone().detach()
        hpc_scatter = HPCScatterConnection(B, M, N, H, W, scatter_type)

        if use_cuda:
            ori_input = ori_input.cuda()
            ori_location = ori_location.cuda()
            ori_scatter = ori_scatter.cuda()

            hpc_input = hpc_input.cuda()
            hpc_location = hpc_location.cuda()
            hpc_scatter = hpc_scatter.cuda()

        for i in range(times):
            t = time.time()
            ori_input.requires_grad_(True)
            ori_output = ori_scatter(ori_input, (H, W), ori_location)
            ori_loss = ori_output * ori_output
            ori_loss = ori_loss.mean()
            ori_loss.backward()
            if use_cuda:
                torch.cuda.synchronize()
            print('epoch: {}, original scatter type {} cost time: {}'.format(i, scatter_type, time.time() - t))

        for i in range(times):
            t = time.time()
            hpc_input.requires_grad_(True)
            hpc_output = hpc_scatter(hpc_input, hpc_location)
            hpc_loss = hpc_output * hpc_output
            hpc_loss = hpc_loss.mean()
            hpc_loss.backward()
            if use_cuda:
                torch.cuda.synchronize()
            print('epoch: {}, hpc scatter type {} cost time: {}'.format(i, scatter_type, time.time() - t))


if __name__ == '__main__':
    print("target problem: B = {}, M = {}, N = {}, H = {}, W = {}".format(B, M, N, H, W))
    print("================run scatter validation test================")
    scatter_val()
    print("================run scatter performance test================")
    scatter_perf()
