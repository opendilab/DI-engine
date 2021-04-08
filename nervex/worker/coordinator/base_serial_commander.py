from collections import namedtuple


class BaseSerialCommander(object):
    r"""
    Overview:
        Base serial commander class.
    Interface:
        __init__, step
    Property:
        policy
    """

    def __init__(
            self,
            cfg: dict,
            learner: 'BaseLearner',  # noqa
            actor: 'BaseSerialActor',  # noqa
            evaluator: 'BaseSerialEvaluator',  # noqa
            replay_buffer: 'BufferManager',  # noqa
            policy: namedtuple = None,
    ) -> None:
        r"""
        Overview:
            Init the BaseSerialCommander
        Arguments:
            - cfg (:obj:`dict`): the config of commander
            - learner (:obj:`BaseLearner`): the learner
            - actor (:obj:`BaseSerialActor`): the actor
            - evaluator (:obj:`BaseSerialEvaluator`): the evaluator
            - replay_buffer (:obj:`BufferManager`): the buffer
        """
        self._cfg = cfg
        self._learner = learner
        self._actor = actor
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
        actor_info = self._actor.actor_info
        self._info.update(learn_info)
        self._info.update(actor_info)
        # update setting
        collect_setting = self._policy.get_setting_collect(self._info)
        self._actor.policy.set_setting('collect', collect_setting)

    @property
    def policy(self) -> 'Policy':  # noqa
        return self._policy

    @policy.setter
    def policy(self, _policy: 'Policy') -> None:  # noqa
        self._policy = _policy
