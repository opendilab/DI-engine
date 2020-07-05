from collections import namedtuple


class EnvElement(object):
    info = namedtuple('EnvElementInfo', ['shape', 'value', 'to_agent_processor', 'from_agent_processor'])
    _instance = None
    _name = 'EnvElement'

    def __init__(self) -> None:
        # placeholder
        # self._shape = 4
        # self._value = {'min': 0, 'max': 1, 'dtype': 'float'}
        # self._to_agent_processor = lambda x: x
        # self._from_agent_processor = None
        assert self._check(), 'this class {} is not a legal subclass of EnvElement'.format(self.__class__)

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # after python3.3, user don't need to pass the extra arguments to the `object` method which is overrided
            cls._instance = object.__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return '{}: {}'.format(self.name, self._details())

    def _details(self) -> str:
        return "placeholder"

    def _check(self) -> bool:
        return all[hasattr(self, '_shape'),
                   hasattr(self, '_value'),
                   hasattr(self, '_to_agent_processer'),
                   hasattr(self, '_from_agent_processor'), ]

    @property
    def info(self) -> 'EnvElement.info':
        return self.info(
            shape=self._shape,
            value=self._value,
            to_agent_processor=self._to_agent_processer,
            from_agent_processor=self._from_agent_processor
        )
