from sc2learner.api import LearnerCommunicationHelper
from .base_learner import BaseLearner


class BaseRLLearner(BaseLearner, LearnerCommunicationHelper):
    _name = "BaseRLLearner"

    def __init__(self, cfg):
        # helper must be initialized first
        LearnerCommunicationHelper.__init__(self, cfg)
        BaseLearner.__init__(self, cfg)
