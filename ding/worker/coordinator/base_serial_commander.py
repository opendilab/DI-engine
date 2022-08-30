from collections import namedtuple
from easydict import EasyDict
import copy


class BaseSerialCommander(object):
    r"""
    Overview:
        Base serial commander class.
    Interface:
        __init__, step
    Property:
        policy
    """

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    config = {}

    def __init__(
            self,
            cfg: dict,
            learner: 'BaseLearner',  # noqa
            collector: 'BaseSerialCollector',  # noqa
            evaluator: 'InteractionSerialEvaluator',  # noqa
            replay_buffer: 'IBuffer',  # noqa
            policy: namedtuple = None,
    ) -> None:
        r"""
        Overview:
            Init the BaseSerialCommander
        Arguments:
            - cfg (:obj:`dict`): the config of commander
            - learner (:obj:`BaseLearner`): the learner
            - collector (:obj:`BaseSerialCollector`): the collector
            - evaluator (:obj:`InteractionSerialEvaluator`): the evaluator
            - replay_buffer (:obj:`IBuffer`): the buffer
        """
        self._cfg = cfg
        self._learner = learner
        self._collector = collector
        self._evaluator = evaluator
        self._replay_buffer = replay_buffer
        self._info = {}
        if policy is not None:
            self.policy = policy

    def step(self) -> None:
        r"""
        Overview:
            Step the commander
        """
        # Update info
        learn_info = self._learner.learn_info
        collector_info = {'envstep': self._collector.envstep}
        self._info.update(learn_info)
        self._info.update(collector_info)
        # update kwargs
        collect_kwargs = self._policy.get_setting_collect(self._info)
        return collect_kwargs

    @property
    def policy(self) -> 'Policy':  # noqa
        return self._policy

    @policy.setter
    def policy(self, _policy: 'Policy') -> None:  # noqa
        self._policy = _policy
