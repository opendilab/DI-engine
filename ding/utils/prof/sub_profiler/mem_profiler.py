import copy
import os
import time
import warnings
from collections import OrderedDict

import torch

try:
    from prettytable import PrettyTable
except ImportError:
    raise Exception("Please pip install prettytable")

version = 'pytorch:' + torch.__version__
use_cuda = torch.cuda.is_available()

N = 1024 * 1024


def get_device():
    if use_cuda:
        return torch.cuda.current_device()
    else:
        return 0


class MemHook(object):

    def __init__(self, total_iter=5, threshold=0, reduction='sum', devices=None, exit=True, compare=False):
        self.total_iter = total_iter
        self.threshold = threshold
        self.reduction = reduction
        self.cur_iter = 0
        self.devices = [devices] if devices else [get_device()]
        self._exit = exit
        self.root = os.path.join('.', 'data')
        self.root = os.path.join(self.root, 'model_prec')
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        self.bt_depth = 6

        filename = f'data_memory{get_device()}.pth'
        self.filepath = os.path.join(self.root, filename)
        self.compare = compare
        if self.compare and not os.path.exists(self.filepath):
            warnings.warn('Reproduct data in {}'.format(version))
            self.compare = False
        self.pre_mem = {}
        self.cur_mem = {}
        self.curr = {}
        self.handles = {}
        self.first_layer = None
        self.start = False

    def pre_hook(self, layer_name, tag='forward'):

        def inner_hook(module, tops):
            self.pre_mem[layer_name + tag + 'pre'] = dict(
                cur_alloc=torch.cuda.memory_allocated(),
                max_alloc=torch.cuda.max_memory_allocated(),
            )

        return inner_hook

    def hook(self, layer_name, tag='forward'):

        def inner_hook(module, tops, bottoms):
            if tag == 'forward' and isinstance(bottoms, dict)\
                    or bottoms is None:
                try:
                    var = bottoms
                    while not isinstance(var, torch.Tensor):
                        if isinstance(var, dict):
                            var = next(v for v in var.values() if isinstance(v, torch.Tensor))
                        else:
                            var = var[0]
                except Exception:
                    print("{} not supported backward hook".format(layer_name))
                    self.remove_handle(layer_name + 'backward')
            key = layer_name + tag
            self.before_hook(key)
            self.cur_mem[layer_name + tag] = dict(
                cur_alloc=torch.cuda.memory_allocated(),
                max_alloc=torch.cuda.max_memory_allocated(),
            )
            self.after_hook(key)

        return inner_hook

    def register_hook(self, model):
        for layer_name, module in model.named_modules():
            if layer_name.split('.') == '':
                pass
            layer_name = layer_name + '.'
            self.curr[layer_name + 'forward'] = 0
            self.curr[layer_name + 'backward'] = 0
            self.handles[layer_name + 'forward'] = \
                module.register_forward_hook(self.hook(layer_name))
            self.handles[layer_name + 'backward'] = \
                module.register_backward_hook(
                    self.hook(layer_name, tag='backward'))
            self.handles[layer_name + 'forwardpre'] = \
                module.register_forward_pre_hook(self.pre_hook(layer_name))
            self.handles

        return model

    def remove_handle(self, key=None):
        if key is not None:
            assert key in self.handles, "{} not in this model".format(key)
            self.handles[key].remove()
        else:
            for k, handle in self.handles.items():
                handle.remove()

    def before_hook(self, key):
        if self.first_layer is None:
            self.first_layer = key
        if key == self.first_layer:
            if not self.start or get_device() not in self.devices:
                self.start = True
            self.cur_iter += 1

    def after_hook(self, key):
        if key != self.first_layer or self.cur_iter < self.total_iter:
            return
        else:
            if self.compare:
                torch.cuda.synchronize()
                time.sleep(get_device() * len(self.cur_mem))
                exp_version, exp_pre, exp_cur = torch.load(self.filepath)
                tb_forward = PrettyTable(float_format="1.1")
                tb_forward.field_names = [
                    "Layer name", "cur Increment(MB)", "exp Increment(MB)", "cur All alloc(MB)", "exp All alloc(MB)"
                ]
                tb_backward = PrettyTable(float_format="1.111")
                tb_backward.field_names = [
                    "Layer name", "cur All alloc(MB)", "exp All alloc(MB)", "cur Max alloc(MB)", "exp Max alloc(MB)"
                ]
                ALLO, MAX = 'cur_alloc', 'max_alloc'
                fmt = "{:.2f}".format
                for k, v in self.cur_mem.items():
                    if 'backward' not in k:
                        tb_forward.add_row(
                            [
                                k,
                                fmt((v[ALLO] - self.pre_mem[k + 'pre'][ALLO]) / N),
                                fmt((exp_cur[k][ALLO] - exp_pre[k + 'pre'][ALLO]) / N),
                                fmt(v[ALLO] / N),
                                fmt(exp_cur[k][ALLO] / N)
                            ]
                        )
                    else:
                        tb_backward.add_row(
                            [k,
                             fmt(v[ALLO] / N),
                             fmt(exp_cur[k][ALLO] / N),
                             fmt(v[MAX] / N),
                             fmt(exp_cur[k][MAX] / N)]
                        )
                print("=========Compare {} with {} Device {}========".format(version, exp_version, get_device()))
                print(tb_forward)
                print(tb_backward)

            self._on_exit()
            self.remove_handle()
            if self._exit:
                import sys
                sys.exit()

    def _on_exit(self):
        if self.compare:
            print('Finished comparision')
        else:
            torch.save([version, self.pre_mem, self.cur_mem], self.filepath)
            print('Finished data saving')
