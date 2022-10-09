from __future__ import print_function
from dizoo.beergame.envs.clBeergame import *
from .utilities import *
import numpy as np
import random
from .config import get_config, update_config
import gym


class BeerGame():

    def __init__(self, role, agentTypes):
        self._cfg, unparsed = get_config()
        self._role = role
        # prepare loggers and directories
        prepare_dirs_and_logger(self._cfg)
        self._cfg = update_config(self._cfg)

        if agentTypes == 'bs':
            self._cfg.agentTypes = ["bs", "bs", "bs", "bs"]
        elif agentTypes == 'Strm':
            self._cfg.agentTypes = ["Strm", "Strm", "Strm", "Strm"]
        self._cfg.agentTypes[role] = "srdqn"

        # get the address of data
        if self._cfg.observation_data:
            adsr = 'data/demandTr-obs-'
        elif self._cfg.demandDistribution == 3:
            if self._cfg.scaled:
                adsr = 'data/basket_data/scaled'
            else:
                adsr = 'data/basket_data'
        elif self._cfg.demandDistribution == 4:
            if self._cfg.scaled:
                adsr = 'data/forecast_data/scaled'
            else:
                adsr = 'data/forecast_data'
        else:
            adsr = 'data/demandTr'

        # load demands
        # demandTr = np.load('demandTr'+str(self._cfg.demandDistribution)+'-'+str(self._cfg.demandUp)+'.npy')
        if self._cfg.demandDistribution == 0:
            direc = os.path.realpath(
                adsr + str(self._cfg.demandDistribution) + '-' + str(self._cfg.demandUp) + '-' +
                str(self._cfg.maxEpisodesTrain) + '.npy'
            )
            if not os.path.exists(direc):
                direc = os.path.realpath(
                    adsr + str(self._cfg.demandDistribution) + '-' + str(self._cfg.demandUp) + '.npy'
                )
        elif self._cfg.demandDistribution == 1:
            direc = os.path.realpath(
                adsr + str(self._cfg.demandDistribution) + '-' + str(int(self._cfg.demandMu)) + '-' +
                str(int(self._cfg.demandSigma)) + '.npy'
            )
        elif self._cfg.demandDistribution == 2:
            direc = os.path.realpath(adsr + str(self._cfg.demandDistribution) + '.npy')
        elif self._cfg.demandDistribution == 3:
            direc = os.path.realpath(adsr + '/demandTr-' + str(self._cfg.data_id) + '.npy')
        elif self._cfg.demandDistribution == 4:
            direc = os.path.realpath(adsr + '/demandTr-' + str(self._cfg.data_id) + '.npy')
        self._demandTr = np.load(direc)
        print("loaded training set=", direc)
        if self._cfg.demandDistribution == 0:
            direc = os.path.realpath(
                'data/demandTs' + str(self._cfg.demandDistribution) + '-' + str(self._cfg.demandUp) + '-' +
                str(self._cfg.maxEpisodesTrain) + '.npy'
            )
            if not os.path.exists(direc):
                direc = os.path.realpath(
                    'data/demandTs' + str(self._cfg.demandDistribution) + '-' + str(self._cfg.demandUp) + '.npy'
                )
        elif self._cfg.demandDistribution == 1:
            direc = os.path.realpath(
                'data/demandTs' + str(self._cfg.demandDistribution) + '-' + str(int(self._cfg.demandMu)) + '-' +
                str(int(self._cfg.demandSigma)) + '.npy'
            )
        elif self._cfg.demandDistribution == 2:
            direc = os.path.realpath('data/demandTs' + str(self._cfg.demandDistribution) + '.npy')
        elif self._cfg.demandDistribution == 3:
            direc = os.path.realpath(adsr + '/demandTs-' + str(self._cfg.data_id) + '.npy')
            direcVl = os.path.realpath(adsr + '/demandVl-' + str(self._cfg.data_id) + '.npy')
            demandVl = np.load(direcVl)
        elif self._cfg.demandDistribution == 4:
            direc = os.path.realpath(adsr + '/demandTs-' + str(self._cfg.data_id) + '.npy')
            direcVl = os.path.realpath(adsr + '/demandVl-' + str(self._cfg.data_id) + '.npy')
            demandVl = np.load(direcVl)
        demandTs = np.load(direc)
        print("loaded test set=", direc)

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
        self._totRew, self._cumReward = self._env.distTotReward(self._role)
        obs = [i for item in self._env.players[self._role].currentState for i in item]
        return obs

    def seed(self, seed):
        self._seed = seed
        np.random.seed(self._seed)

    def close(self):
        pass

    def step(self, action):
        self._env.handelAction(action)  # action is logit, for example: action=[0,1,0,0,0], only input
        self._env.next()
        newstate = np.append(
            self._env.players[self._role].currentState[1:, :], [self._env.players[self._role].nextObservation], axis=0
        )
        self._env.players[self._role].currentState = newstate
        obs = [i for item in newstate for i in item]
        rew = self._env.players[self._role].curReward + (self._cfg.distCoeff /
                                                         3) * ((self._totRew - self._cumReward) / (self._env.T))
        done = (self._env.curTime == self._env.T)
        info = {}

        return obs, rew, done, info
