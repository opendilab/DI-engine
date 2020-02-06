import numpy as np
import os
import torch
import linklink as link


def get_rank():
    return link.get_rank()


def get_world_size():
    return link.get_world_size()


def allreduce(data):
    link.allreduce(data)


def distributed_mode(func):

    def wrapper(*args, **kwargs):
        dist_init()
        func(*args, **kwargs)
        dist_finalize()

    return wrapper


def dist_init(method='slurm', device_id=0):
    if method == 'slurm':
        proc_id = int(os.environ['SLURM_PROCID'])
        # ntasks = int(os.environ['SLURM_NTASKS'])
        # node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(proc_id % num_gpus)
    elif method == 'single_node':
        torch.cuda.set_device(device_id)

    link.initialize()
    world_size = link.get_world_size()
    rank = link.get_rank()

    return rank, world_size


def dist_finalize():
    link.finalize()


def simple_group_split(world_size, rank, num_groups):
    groups = []
    rank_list = np.split(np.arange(world_size), num_groups)
    rank_list = [list(map(int, x)) for x in rank_list]
    for i in range(num_groups):
        groups.append(link.new_group(rank_list[i]))
    group_size = world_size // num_groups
    return groups[rank//group_size]


class DistModule(torch.nn.Module):
    def __init__(self, module, sync=True):
        super(DistModule, self).__init__()
        self.module = module
        self.broadcast_params()

        self.sync = sync
        if not sync:
            self._grad_accs = []
            self._register_hooks()
        self._create_grad()

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def _register_hooks(self):
        for i, (name, p) in enumerate(self.named_parameters()):
            if p.requires_grad:
                p_tmp = p.expand_as(p)
                grad_acc = p_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(self._make_hook(name, p, i))
                self._grad_accs.append(grad_acc)

    def _make_hook(self, name, p, i):
        def hook(*ignore):
            link.allreduce_async(name, p.grad.data)
        return hook

    def sync_gradients(self):
        """ average gradients """
        if self.sync and link.get_world_size() > 1:
            for name, param in self.module.named_parameters():
                if param.requires_grad:
                    link.allreduce(param.grad.data)
        else:
            link.synchronize()

    def broadcast_params(self):
        """ broadcast model parameters """
        for name, param in self.module.state_dict().items():
            link.broadcast(param, 0)

    def _create_grad(self):
        for name, param in self.module.named_parameters():
            setattr(param, 'grad', torch.zeros_like(param))
