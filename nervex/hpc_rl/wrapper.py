import importlib
from functools import wraps
import nervex

hpc_fns = {}


def register_runtime_fn(fn_name, runtime_name, shape):
    fn_name_mapping = {
        'gae': ['hpc_rl.loss.gae', 'HPCGAE'],
    }
    fn_str = fn_name_mapping[fn_name]
    cls = getattr(importlib.import_module(fn_str[0]), fn_str[1])
    hpc_fn = cls(*shape).cuda()
    hpc_fns[runtime_name] = hpc_fn
    return hpc_fn


def hpc_wrapper(shape_fn=None, namedtuple_data=False):

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
                if namedtuple_data:
                    data = args[0]  # args[0] is a namedtuple
                    return hpc_fn(*data, *args[1:], **kwargs)
                else:
                    return hpc_fn(*args, **kwargs)
            else:
                return fn(*args, **kwargs)

        return wrapper

    return decorate
