import os
import inspect
from collections import OrderedDict

_innest_error = True

_NERVEX_REG_TRACE_IS_ON = os.environ.get('NERVEXREGTRACE', 'OFF').upper() == 'ON'


def lowercase(name):
    return ''.join([letter if letter.islower() else '_' + letter for letter in list(name)])


class Registry(dict):
    """
    A helper class for managing registering modules, it extends a dictionary
    and provides a register functions.

    Eg. creeting a registry:
        some_registry = Registry({"default": default_module})

    There're two ways of registering new modules:
    1): normal way is just calling register function:
        def foo():
            ...
        some_registry.register("foo_module", foo)
    2): used as decorator when declaring the module:
        @some_registry.register("foo_module")
        @some_registry.register("foo_modeul_nickname")
        def foo():
            ...

    Access of module is just like using a dictionary, eg:
        f = some_registry["foo_module"]
    """

    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)
        self.__trace__ = dict()

    def register(self, module_name=None, module=None):
        if _NERVEX_REG_TRACE_IS_ON:
            frame = inspect.stack()[1][0]
            info = inspect.getframeinfo(frame)
            filename = info.filename
            lineno = info.lineno
        # used as function call
        if module is not None:
            assert module_name is not None
            Registry._register_generic(self, module_name, module)
            if _NERVEX_REG_TRACE_IS_ON:
                self.__trace__[module_name] = (filename, lineno)
            return

        # used as decorator
        def register_fn(fn):
            if module_name is None:
                name = fn.__name__
            else:
                name = module_name
            Registry._register_generic(self, name, fn)
            if _NERVEX_REG_TRACE_IS_ON:
                self.__trace__[name] = (filename, lineno)
            return fn

        return register_fn

    @staticmethod
    def _register_generic(module_dict, module_name, module):
        assert module_name not in module_dict, module_name
        module_dict[module_name] = module

    def get(self, module_name):
        return self[module_name]

    def build(self, obj_type, obj_cfg, **obj_kwargs):

        try:
            build_fn = self[obj_type]
            return build_fn(obj_cfg, **obj_kwargs)
        except Exception as e:
            global _innest_error
            if _innest_error:
                argspec = inspect.getfullargspec(build_fn)
                message = 'for {}(alias={})'.format(build_fn, obj_type)
                message += '\nExpected args are:{}'.format(argspec)
                message += '\nGiven args are:{}'.format(argspec, obj_kwargs.keys())
                message += '\nGiven args details are:{}'.format(argspec, obj_kwargs)
                _innest_error = False
            raise e

    def query(self):
        return self.keys()

    def query_details(self, aliases=None):
        assert _NERVEX_REG_TRACE_IS_ON, "please exec 'export NERVEXREGTRACE=ON' first"
        if aliases is None:
            aliases = self.keys()
        return OrderedDict((alias, self.__trace__[alias]) for alias in aliases)
