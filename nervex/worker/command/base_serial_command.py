class BaseSerialCommand(object):

    def __init__(
            self, cfg: dict, learner: 'BaseLearner', actor: 'BaseSerialActor', evaluator: 'BaseSerialEvaluator',
            replay_buffer: 'ReplayBuffer'
    ) -> None:
        self._cfg = cfg
        self._learner = learner
        self._actor = actor
        self._evaluator = evaluator
        self._replay_buffer = replay_buffer
        self._info = {}

    def step(self) -> None:
        # update info
        learner_info = self._learner.get_current_info()
        self._info.update(learner_info)
        # update setting
        collect_setting = self._policy.get_setting_collect(self._info)
        # set setting
        self._actor.policy.set_setting('collect', collect_setting)

    @property
    def policy(self) -> 'Policy':  # noqa
        return self._policy

    @policy.setter
    def policy(self, _policy) -> None:
        self._policy = _policy
