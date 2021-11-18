import inspect
import os
from collections import OrderedDict
from typing import Optional, Iterable, Callable

_innest_error = True

_DI_ENGINE_REG_TRACE_IS_ON = os.environ.get('DIENGINEREGTRACE', 'OFF').upper() == 'ON'


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

    def __init__(self, *args, **kwargs) -> None:
        super(Registry, self).__init__(*args, **kwargs)
        self.__trace__ = dict()

    def register(
            self,
            module_name: Optional[str] = None,
            module: Optional[Callable] = None,
            force_overwrite: bool = False
    ) -> Callable:
        if _DI_ENGINE_REG_TRACE_IS_ON:
            frame = inspect.stack()[1][0]
            info = inspect.getframeinfo(frame)
            filename = info.filename
            lineno = info.lineno
        # used as function call
        if module is not None:
            assert module_name is not None
            Registry._register_generic(self, module_name, module, force_overwrite)
            if _DI_ENGINE_REG_TRACE_IS_ON:
                self.__trace__[module_name] = (filename, lineno)
            return

        # used as decorator
        def register_fn(fn: Callable) -> Callable:
            if module_name is None:
                name = fn.__name__
            else:
                name = module_name
            Registry._register_generic(self, name, fn, force_overwrite)
            if _DI_ENGINE_REG_TRACE_IS_ON:
                self.__trace__[name] = (filename, lineno)
            return fn

        return register_fn

    @staticmethod
    def _register_generic(module_dict: dict, module_name: str, module: Callable, force_overwrite: bool = False) -> None:
        if not force_overwrite:
            assert module_name not in module_dict, module_name
        module_dict[module_name] = module

    def get(self, module_name: str) -> Callable:
        return self[module_name]

    def build(self, obj_type: str, *obj_args, **obj_kwargs) -> object:
        try:
            build_fn = self[obj_type]
            return build_fn(*obj_args, **obj_kwargs)
        except Exception as e:
            # get build_fn fail
            if isinstance(e, KeyError):
                raise KeyError("not support buildable-object type: {}".format(obj_type))
            # build_fn execution fail
            global _innest_error
            if _innest_error:
                argspec = inspect.getfullargspec(build_fn)
                message = 'for {}(alias={})'.format(build_fn, obj_type)
                message += '\nExpected args are:{}'.format(argspec)
                message += '\nGiven args are:{}/{}'.format(argspec, obj_kwargs.keys())
                message += '\nGiven args details are:{}/{}'.format(argspec, obj_kwargs)
                _innest_error = False
            raise e

    def query(self) -> Iterable:
        return self.keys()

    def query_details(self, aliases: Optional[Iterable] = None) -> OrderedDict:
        assert _DI_ENGINE_REG_TRACE_IS_ON, "please exec 'export DIENGINEREGTRACE=ON' first"
        if aliases is None:
            aliases = self.keys()
        return OrderedDict((alias, self.__trace__[alias]) for alias in aliases)
