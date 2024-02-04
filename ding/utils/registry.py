import inspect
import os
from collections import OrderedDict
from typing import Optional, Iterable, Callable

_innest_error = True

_DI_ENGINE_REG_TRACE_IS_ON = os.environ.get('DIENGINEREGTRACE', 'OFF').upper() == 'ON'


class Registry(dict):
    """
    Overview:
        A helper class for managing registering modules, it extends a dictionary
        and provides a register functions.
    Interfaces:
        ``__init__``, ``register``, ``get``, ``build``, ``query``, ``query_details``
    Examples (creating):
        >>> some_registry = Registry({"default": default_module})

    Examples (registering: normal way):
        >>> def foo():
        >>>     ...
        >>> some_registry.register("foo_module", foo)

    Examples (registering: decorator way):
        >>> @some_registry.register("foo_module")
        >>> @some_registry.register("foo_modeul_nickname")
        >>> def foo():
        >>>     ...

    Examples (accessing):
        >>> f = some_registry["foo_module"]
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Overview:
            Initialize the Registry object.
        Arguments:
            - args (:obj:`Tuple`): The arguments passed to the ``__init__`` function of the parent class, \
                dict.
            - kwargs (:obj:`Dict`): The keyword arguments passed to the ``__init__`` function of the parent class, \
                dict.
        """

        super(Registry, self).__init__(*args, **kwargs)
        self.__trace__ = dict()

    def register(
            self,
            module_name: Optional[str] = None,
            module: Optional[Callable] = None,
            force_overwrite: bool = False
    ) -> Callable:
        """
        Overview:
            Register the module.
        Arguments:
            - module_name (:obj:`Optional[str]`): The name of the module.
            - module (:obj:`Optional[Callable]`): The module to be registered.
            - force_overwrite (:obj:`bool`): Whether to overwrite the module with the same name.
        """

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
        """
        Overview:
            Register the module.
        Arguments:
            - module_dict (:obj:`dict`): The dict to store the module.
            - module_name (:obj:`str`): The name of the module.
            - module (:obj:`Callable`): The module to be registered.
            - force_overwrite (:obj:`bool`): Whether to overwrite the module with the same name.
        """

        if not force_overwrite:
            assert module_name not in module_dict, module_name
        module_dict[module_name] = module

    def get(self, module_name: str) -> Callable:
        """
        Overview:
            Get the module.
        Arguments:
            - module_name (:obj:`str`): The name of the module.
        """

        return self[module_name]

    def build(self, obj_type: str, *obj_args, **obj_kwargs) -> object:
        """
        Overview:
            Build the object.
        Arguments:
            - obj_type (:obj:`str`): The type of the object.
            - obj_args (:obj:`Tuple`): The arguments passed to the object.
            - obj_kwargs (:obj:`Dict`): The keyword arguments passed to the object.
        """

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
                message = 'Hint: for {}(alias={})'.format(build_fn, obj_type)
                message += '\n\nExpected args are:\n {}\nGiven arguments keys are:\n{}\n'.format(
                    argspec, obj_kwargs.keys()
                )
                print(message)
                _innest_error = False
            raise e

    def query(self) -> Iterable:
        """
        Overview:
            all registered module names.
        """

        return self.keys()

    def query_details(self, aliases: Optional[Iterable] = None) -> OrderedDict:
        """
        Overview:
            Get the details of the registered modules.
        Arguments:
            - aliases (:obj:`Optional[Iterable]`): The aliases of the modules.
        """

        assert _DI_ENGINE_REG_TRACE_IS_ON, "please exec 'export DIENGINEREGTRACE=ON' first"
        if aliases is None:
            aliases = self.keys()
        return OrderedDict((alias, self.__trace__[alias]) for alias in aliases)
