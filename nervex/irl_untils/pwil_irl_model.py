from .base_reward_estimate import BaseRewardModel
import pickle


class PwilRewardModel(BaseRewardModel):

    def __init__(self, config: dict) -> None:
        super(PwilRewardModel, self).__init__()
        self.config = config
        self.expert_data = []

    def load_expert_data(self) -> None:
        with open(self.config['expert_data_path'], 'rb') as f:
            self.expert_data = pickle.load(f)

    def launch(self) -> None:
        self.load_expert_data()

    def estimate(self, s, a) -> None:
        pass

    def train(self) -> None:
        pass
