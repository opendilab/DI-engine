from __future__ import print_function
from dizoo.beergame.envs import clBeerGame
from torch import Tensor
import numpy as np
import random
from .utils import get_config, update_config
import gym
import os
from typing import Optional


class BeerGame():

    def __init__(self, role: int, agent_type: str, demandDistribution: int) -> None:
        self._cfg, unparsed = get_config()
        self._role = role
        # prepare loggers and directories
        # prepare_dirs_and_logger(self._cfg)
        self._cfg = update_config(self._cfg)

        # set agent type
        if agent_type == 'bs':
            self._cfg.agentTypes = ["bs", "bs", "bs", "bs"]
        elif agent_type == 'Strm':
            self._cfg.agentTypes = ["Strm", "Strm", "Strm", "Strm"]
        self._cfg.agentTypes[role] = "srdqn"

        self._cfg.demandDistribution = demandDistribution

        # load demands:0=uniform, 1=normal distribution, 2=the sequence of 4,4,4,4,8,..., 3= basket data, 4= forecast data
        if self._cfg.observation_data:
            adsr = 'data/demandTr-obs-'
        elif self._cfg.demandDistribution == 3:
            if self._cfg.scaled:
                adsr = 'data/basket_data/scaled'
            else:
                adsr = 'data/basket_data'
            direc = os.path.realpath(adsr + '/demandTr-' + str(self._cfg.data_id) + '.npy')
            self._demandTr = np.load(direc)
            print("loaded training set=", direc)
        elif self._cfg.demandDistribution == 4:
            if self._cfg.scaled:
                adsr = 'data/forecast_data/scaled'
            else:
                adsr = 'data/forecast_data'
            direc = os.path.realpath(adsr + '/demandTr-' + str(self._cfg.data_id) + '.npy')
            self._demandTr = np.load(direc)
            print("loaded training set=", direc)
        else:
            if self._cfg.demandDistribution == 0:  # uniform
                self._demandTr = np.random.randint(0, self._cfg.demandUp, size=[self._cfg.demandSize, self._cfg.TUp])
            elif self._cfg.demandDistribution == 1:  # normal distribution
                self._demandTr = np.round(
                    np.random.normal(
                        self._cfg.demandMu, self._cfg.demandSigma, size=[self._cfg.demandSize, self._cfg.TUp]
                    )
                ).astype(int)
            elif self._cfg.demandDistribution == 2:  # the sequence of 4,4,4,4,8,...
                self._demandTr = np.concatenate(
                    (4 * np.ones((self._cfg.demandSize, 4)), 8 * np.ones((self._cfg.demandSize, 98))), axis=1
                ).astype(int)

        # initilize an instance of Beergame
        self._env = clBeerGame(self._cfg)
        self.observation_space = gym.spaces.Box(
            low=float("-inf"),
            high=float("inf"),
            shape=(self._cfg.stateDim * self._cfg.multPerdInpt, ),
            dtype=np.float32
        )  # state_space = state_dim * m (considering the reward delay)
        self.action_space = gym.spaces.Discrete(self._cfg.actionListLen)  # length of action list
        self.reward_space = gym.spaces.Box(low=float("-inf"), high=float("inf"), shape=(1, ), dtype=np.float32)

        # get the length of the demand.
        self._demand_len = np.shape(self._demandTr)[0]

    def reset(self):
        self._env.resetGame(demand=self._demandTr[random.randint(0, self._demand_len - 1)])
        obs = [i for item in self._env.players[self._role].currentState for i in item]
        return obs

    def seed(self, seed: int) -> None:
        self._seed = seed
        np.random.seed(self._seed)

    def close(self) -> None:
        pass

    def step(self, action: np.ndarray):
        self._env.handelAction(action)
        self._env.next()
        newstate = np.append(
            self._env.players[self._role].currentState[1:, :], [self._env.players[self._role].nextObservation], axis=0
        )
        self._env.players[self._role].currentState = newstate
        obs = [i for item in newstate for i in item]
        rew = self._env.players[self._role].curReward
        done = (self._env.curTime == self._env.T)
        info = {}
        return obs, rew, done, info

    def reward_shaping(self, reward: Tensor) -> Tensor:
        self._totRew, self._cumReward = self._env.distTotReward(self._role)
        reward += (self._cfg.distCoeff / 3) * ((self._totRew - self._cumReward) / (self._env.T))
        return reward

    def enable_save_figure(self, figure_path: Optional[str] = None) -> None:
        self._cfg.ifSaveFigure = True
        if figure_path is None:
            figure_path = './'
        self._cfg.figure_dir = figure_path
        self._env.doTestMid(self._demandTr[random.randint(0, self._demand_len - 1)])
