import importlib
from functools import wraps
import nervex

hpc_fns = {}


def register_runtime_fn(fn_name, runtime_name, shape):
    fn_name_mapping = {
        'gae': ['hpc_rl.loss.gae', 'HPCGAE'],
        'dist_nstep_td_error': ['hpc_rl.loss.td', 'HPCDNTD'],
        'LSTM': ['hpc_rl.network.rnn', 'HPCLSTM'],
        'ppo_error': ['hpc_rl.loss.ppo', 'HPCPPO'],
        'q_nstep_td_error': ['hpc_rl.loss.td', 'HPCQNTD'],
        'q_nstep_td_error_with_rescale': ['hpc_rl.loss.td', 'HPCQNTDRescale'],
        'ScatterConnection': ['hpc_rl.network.scatter_connection', 'HPCScatterConnection'],
        'td_lambda_error': ['hpc_rl.loss.td', 'HPCTDLambda'],
        'upgo_loss': ['hpc_rl.loss.upgo', 'HPCUPGO'],
        'vtrace_error': ['hpc_rl.loss.vtrace', 'HPCVtrace'],
    }
    fn_str = fn_name_mapping[fn_name]
    cls = getattr(importlib.import_module(fn_str[0]), fn_str[1])
    hpc_fn = cls(*shape).cuda()
    hpc_fns[runtime_name] = hpc_fn
    return hpc_fn


def hpc_wrapper(shape_fn=None, namedtuple_data=False, include_args=None, include_kwargs=[]):

    def decorate(fn):

        @wraps(fn)
        def wrapper(*args, **kwargs):
            if nervex.enable_hpc_rl:
                shape = shape_fn(args)
                runtime_name = '_'.join([fn.__name__] + [str(s) for s in shape])
                if runtime_name not in hpc_fns:
                    # TODO(nyz) buffer release
                    hpc_fn = register_runtime_fn(fn.__name__, runtime_name, shape)
                else:
                    hpc_fn = hpc_fns[runtime_name]
                if include_args is None:
                    end = len(args)
                else:
                    end = include_args
                clean_kwargs = {}
                for k, v in kwargs.items():
                    if k in include_kwargs:
                        clean_kwargs[k] = v
                if namedtuple_data:
                    data = args[0]  # args[0] is a namedtuple
                    return hpc_fn(*data, *args[1:end], **clean_kwargs)
                else:
                    return hpc_fn(*args, **clean_kwargs)
            else:
                return fn(*args, **kwargs)

        return wrapper

    return decorate
