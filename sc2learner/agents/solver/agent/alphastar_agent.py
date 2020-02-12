from .agent import BaseAgent
from sc2learner.agents.model import build_model
from sc2learner.utils. import to_device


class AlphastarAgent(BaseAgent):

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = build_model(cfg)
        self.model.eval()
        self.use_cuda = cfg.train.use_cuda
        if self.use_cuda:
            self.model = to_device(self.model, 'cuda')

        self.next_state = None

    def reset(self):
        self.next_state = None

    def act(self, obs):
        obs['prev_state'] = self.next_state
        ret = self.model(obs, mode='evaluate')
        actions, self.next_state = ret['actions'], ret['next_state']
        return actions

    def value(self):
        return 0
