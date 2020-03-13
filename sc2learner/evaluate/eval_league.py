import os
import torch


class EvalLeague(object):
    def __init__(self, ):
        self.elo_rating = []
        self.init_league()

    def init_league(self, ):
        raise NotImplementedError

    def save_league(self):
        raise NotImplementedError

    def add_reference(self):
        raise NotImplementedError

    def rating_agent(self, agent):
        raise NotImplementedError
