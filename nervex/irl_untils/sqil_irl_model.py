"""
    @description: the implement of reward model in sqil(https://arxiv.org/abs/1905.11108)
    @author: liuyang
"""

from base_reward_estimate import BaseRewardModel
import pickle


class SqilRewardModel(BaseRewardModel):

    def __init__(self, config: dict):
        super(SqilRewardModel, self).__init__()
        self.config = config
        self.expert_data = []

    def load_expert_data(self) -> None:
        with open(self.config['expert_data_path'], 'rb') as f:
            self.expert_data = pickle.load(f)

    def launch(self) -> None:
        self.load_expert_data()

    def train(self) -> None:
        pass

    def estimate(self, s, a) -> None:
        pass
