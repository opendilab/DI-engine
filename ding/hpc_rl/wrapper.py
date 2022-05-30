import importlib
from ditk import logging
from collections import OrderedDict
from functools import wraps
import ding
'''
Overview:
    `hpc_wrapper` is the wrapper for functions which are supported by hpc. If a function is wrapped by it, we will
    search for its hpc type and return the function implemented by hpc.
    We will use the following code as a sample to introduce `hpc_wrapper`:
    ```
    @hpc_wrapper(shape_fn=shape_fn_dntd, namedtuple_data=True, include_args=[0,1,2,3],
                 include_kwargs=['data', 'gamma', 'v_min', 'v_max'], is_cls_method=False)
    def dist_nstep_td_error(
            data: namedtuple,
            gamma: float,
            v_min: float,
            v_max: float,
            n_atom: int,
            nstep: int = 1,
    ) -> torch.Tensor:
    ...
    ```
Parameters:
    - shape_fn (:obj:`function`): a function which return the shape needed by hpc function. In fact, it returns
        all args that the hpc function needs.
    - nametuple_data (:obj:`bool`): If True, when hpc function is called, it will be called as hpc_function(*nametuple).
        If False, nametuple data will remain its `nametuple` type.
    - include_args (:obj:`list`): a list of index of the args need to be set in hpc function. As shown in the sample,
        include_args=[0,1,2,3], which means `data`, `gamma`, `v_min` and `v_max` will be set in hpc function.
    - include_kwargs (:obj:`list`): a list of key of the kwargs need to be set in hpc function. As shown in the sample,
        include_kwargs=['data', 'gamma', 'v_min', 'v_max'], which means `data`, `gamma`, `v_min` and `v_max` will be
        set in hpc function.
    - is_cls_method (:obj:`bool`): If True, it means the function we wrap is a method of a class. `self` will be put
        into args. We will get rid of `self` in args. Besides, we will use its classname as its fn_name.
        If False, it means the function is a simple method.
Q&A:
    - Q: Is `include_args` and `include_kwargs` need to be set at the same time?
    - A: Yes. `include_args` and `include_kwargs` can deal with all type of input, such as (data, gamma, v_min=v_min,
        v_max=v_max) and (data, gamma, v_min, v_max).
    - Q: What is `hpc_fns`?
    - A: Here we show a normal `hpc_fns`:
         ```
         hpc_fns = {
             'fn_name1': {
                 'runtime_name1': hpc_fn1,
                 'runtime_name2': hpc_fn2,
                 ...
             },
             ...
         }
         ```
         Besides, `per_fn_limit` means the max length of `hpc_fns[fn_name]`. When new function comes, the oldest
         function will be popped from `hpc_fns[fn_name]`.
'''

hpc_fns = {}
per_fn_limit = 3


def register_runtime_fn(fn_name, runtime_name, shape):
    fn_name_mapping = {
        'gae': ['hpc_rll.rl_utils.gae', 'GAE'],
        'dist_nstep_td_error': ['hpc_rll.rl_utils.td', 'DistNStepTD'],
        'LSTM': ['hpc_rll.torch_utils.network.rnn', 'LSTM'],
        'ppo_error': ['hpc_rll.rl_utils.ppo', 'PPO'],
        'q_nstep_td_error': ['hpc_rll.rl_utils.td', 'QNStepTD'],
        'q_nstep_td_error_with_rescale': ['hpc_rll.rl_utils.td', 'QNStepTDRescale'],
        'ScatterConnection': ['hpc_rll.torch_utils.network.scatter_connection', 'ScatterConnection'],
        'td_lambda_error': ['hpc_rll.rl_utils.td', 'TDLambda'],
        'upgo_loss': ['hpc_rll.rl_utils.upgo', 'UPGO'],
        'vtrace_error': ['hpc_rll.rl_utils.vtrace', 'VTrace'],
    }
    fn_str = fn_name_mapping[fn_name]
    cls = getattr(importlib.import_module(fn_str[0]), fn_str[1])
    hpc_fn = cls(*shape).cuda()
    if fn_name not in hpc_fns:
        hpc_fns[fn_name] = OrderedDict()
    hpc_fns[fn_name][runtime_name] = hpc_fn
    while len(hpc_fns[fn_name]) > per_fn_limit:
        hpc_fns[fn_name].popitem(last=False)
    # print(hpc_fns)
    return hpc_fn


def hpc_wrapper(shape_fn=None, namedtuple_data=False, include_args=[], include_kwargs=[], is_cls_method=False):

    def decorate(fn):

        @wraps(fn)
        def wrapper(*args, **kwargs):
            if ding.enable_hpc_rl:
                shape = shape_fn(args, kwargs)
                if is_cls_method:
                    fn_name = args[0].__class__.__name__
                else:
                    fn_name = fn.__name__
                runtime_name = '_'.join([fn_name] + [str(s) for s in shape])
                if fn_name not in hpc_fns or runtime_name not in hpc_fns[fn_name]:
                    hpc_fn = register_runtime_fn(fn_name, runtime_name, shape)
                else:
                    hpc_fn = hpc_fns[fn_name][runtime_name]
                if is_cls_method:
                    args = args[1:]
                clean_args = []
                for i in include_args:
                    if i < len(args):
                        clean_args.append(args[i])
                nouse_args = list(set(list(range(len(args)))).difference(set(include_args)))
                clean_kwargs = {}
                for k, v in kwargs.items():
                    if k in include_kwargs:
                        if k == 'lambda_':
                            k = 'lambda'
                        clean_kwargs[k] = v
                nouse_kwargs = list(set(kwargs.keys()).difference(set(include_kwargs)))
                if len(nouse_args) > 0 or len(nouse_kwargs) > 0:
                    logging.warn(
                        'in {}, index {} of args are dropped, and keys {} of kwargs are dropped.'.format(
                            runtime_name, nouse_args, nouse_kwargs
                        )
                    )
                if namedtuple_data:
                    data = args[0]  # args[0] is a namedtuple
                    return hpc_fn(*data, *clean_args[1:], **clean_kwargs)
                else:
                    return hpc_fn(*clean_args, **clean_kwargs)
            else:
                return fn(*args, **kwargs)

        return wrapper

    return decorate
