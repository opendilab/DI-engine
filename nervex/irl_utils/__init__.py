from .base_reward_estimate import BaseRewardModel, create_irl_model, register_irl_model
from .pdeil_irl_model import PdeilRewardModel
from .gail_irl_model import GailRewardModel
register_irl_model('pdeil', PdeilRewardModel)
register_irl_model('gail', GailRewardModel)
