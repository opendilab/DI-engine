from .ppo_learner import PpoLearner
import os
if 'IS_K8S' not in os.environ:
    # currently we have no support for AS in K8s
    from .alphastar_sl_learner import AlphastarSLLearner
