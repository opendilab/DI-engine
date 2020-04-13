from sc2learner.api import LearnerCommunicationHelper
from .base_learner import Learner


class BaseRLLearner(Learner, LearnerCommunicationHelper):
    _name = "BaseRLLearner"

    def __init__(self, cfg):
        # helper must be initialized first
        LearnerCommunicationHelper.__init__(self, cfg)
        Learner.__init__(self, cfg)
