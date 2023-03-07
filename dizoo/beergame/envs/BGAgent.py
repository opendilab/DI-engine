# Code Reference: https://github.com/OptMLGroup/DeepBeerInventory-RL.
import argparse
import numpy as np


# Here we want to define the agent class for the BeerGame
class Agent(object):
    # initializes the agents with initial values for IL, OO and saves self.agentNum for recognizing the agents.
    def __init__(
            self, agentNum: int, IL: int, AO: int, AS: int, c_h: float, c_p: float, eta: int, compuType: str,
            config: argparse.Namespace
    ) -> None:
        self.agentNum = agentNum
        self.IL = IL  # Inventory level of each agent - changes during the game
        self.OO = 0  # Open order of each agent - changes during the game
        self.ASInitial = AS  # the initial arriving shipment.
        self.ILInitial = IL  # IL at which we start each game with this number
        self.AOInitial = AO  # OO at which we start each game with this number
        self.config = config  # an instance of config is stored inside the class
        self.curState = []  # this function gets the current state of the game
        self.nextState = []
        self.curReward = 0  # the reward observed at the current step
        self.cumReward = 0  # cumulative reward; reset at the beginning of each episode
        self.totRew = 0  # it is reward of all players obtained for the current player.
        self.c_h = c_h  # holding cost
        self.c_p = c_p  # backorder cost
        self.eta = eta  # the total cost regulazer
        self.AS = np.zeros((1, 1))  # arriced shipment
        self.AO = np.zeros((1, 1))  # arrived order
        self.action = 0  # the action at time t
        self.compType = compuType
        # self.compTypeTrain = compuType  # rnd -> random / srdqn-> srdqn / Strm-> formula-Rong2008 / bs -> optimal policy if exists
        # self.compTypeTest = compuType # rnd -> random / srdqn-> srdqn / Strm-> formula-Rong2008 / bs -> optimal policy if exists
        self.alpha_b = self.config.alpha_b[self.agentNum]  # parameters for the formula
        self.betta_b = self.config.betta_b[self.agentNum]  # parameters for the formula
        if self.config.demandDistribution == 0:
            self.a_b = np.mean((self.config.demandUp, self.config.demandLow))  # parameters for the formula
            self.b_b = np.mean((self.config.demandUp, self.config.demandLow)) * (
                np.mean((self.config.leadRecItemLow[self.agentNum], self.config.leadRecItemUp[self.agentNum])) +
                np.mean((self.config.leadRecOrderLow[self.agentNum], self.config.leadRecOrderUp[self.agentNum]))
            )  # parameters for the formula
        elif self.config.demandDistribution == 1 or self.config.demandDistribution == 3 or self.config.demandDistribution == 4:
            self.a_b = self.config.demandMu  # parameters for the formula
            self.b_b = self.config.demandMu * (
                np.mean((self.config.leadRecItemLow[self.agentNum], self.config.leadRecItemUp[self.agentNum])) +
                np.mean((self.config.leadRecOrderLow[self.agentNum], self.config.leadRecOrderUp[self.agentNum]))
            )  # parameters for the formula
        elif self.config.demandDistribution == 2:
            self.a_b = 8  # parameters for the formula
            self.b_b = (3 / 4.) * 8 * (
                np.mean((self.config.leadRecItemLow[self.agentNum], self.config.leadRecItemUp[self.agentNum])) +
                np.mean((self.config.leadRecOrderLow[self.agentNum], self.config.leadRecOrderUp[self.agentNum]))
            )  # parameters for the formula
        elif self.config.demandDistribution == 3:
            self.a_b = 10  # parameters for the formula
            self.b_b = 7 * (
                np.mean((self.config.leadRecItemLow[self.agentNum], self.config.leadRecItemUp[self.agentNum])) +
                np.mean((self.config.leadRecOrderLow[self.agentNum], self.config.leadRecOrderUp[self.agentNum]))
            )  # parameters for the formula
        else:
            raise Exception('The demand distribution is not defined or it is not a valid type.!')

        self.hist = []  # this is used for plotting - keeps the history for only one game
        self.hist2 = []  # this is used for animation usage
        self.srdqnBaseStock = []  # this holds the base stock levels that srdqn has came up with. added on Nov 8, 2017
        self.T = 0
        self.bsBaseStock = 0
        self.init_bsBaseStock = 0
        self.nextObservation = []

        if self.compType == 'srdqn':
            # sets the initial input of the network
            self.currentState = np.stack(
                [self.curState for _ in range(self.config.multPerdInpt)], axis=0
            )  # multPerdInpt observations stacked. each row is an observation

    # reset player information
    def resetPlayer(self, T: int):
        self.IL = self.ILInitial
        self.OO = 0
        self.AS = np.squeeze(
            np.zeros((1, T + max(self.config.leadRecItemUp) + max(self.config.leadRecOrderUp) + 10))
        )  # arriced shipment
        self.AO = np.squeeze(
            np.zeros((1, T + max(self.config.leadRecItemUp) + max(self.config.leadRecOrderUp) + 10))
        )  # arrived order
        if self.agentNum != 0:
            for i in range(self.config.leadRecOrderUp_aux[self.agentNum - 1]):
                self.AO[i] = self.AOInitial[self.agentNum - 1]
        for i in range(self.config.leadRecItemUp[self.agentNum]):
            self.AS[i] = self.ASInitial
        self.curReward = 0  # the reward observed at the current step
        self.cumReward = 0  # cumulative reward; reset at the begining of each episode
        self.action = []
        self.hist = []
        self.hist2 = []
        self.srdqnBaseStock = []  # this holds the base stock levels that srdqn has came up with. added on Nov 8, 2017
        self.T = T
        self.curObservation = self.getCurState(1)  # this function gets the current state of the game
        self.nextObservation = []
        if self.compType == 'srdqn':
            self.currentState = np.stack([self.curObservation for _ in range(self.config.multPerdInpt)], axis=0)

    # updates the IL and OO at time t, after recieving "rec" number of items
    def recieveItems(self, time: int) -> None:
        self.IL = self.IL + self.AS[time]  # inverntory level update
        self.OO = self.OO - self.AS[time]  # invertory in transient update

    # find action Value associated with the action list
    def actionValue(self, curTime: int) -> int:
        if self.config.fixedAction:
            a = self.config.actionList[np.argmax(self.action)]
        else:
            # "d + x" rule
            if self.compType == 'srdqn':
                a = max(0, self.config.actionList[np.argmax(self.action)] * self.config.action_step + self.AO[curTime])
            elif self.compType == 'rnd':
                a = max(0, self.config.actionList[np.argmax(self.action)] + self.AO[curTime])
            else:
                a = max(0, self.config.actionListOpt[np.argmax(self.action)])

        return a

    # getReward returns the reward at the current state
    def getReward(self) -> None:
        # cost (holding + backorder) for one time unit
        self.curReward = (self.c_p * max(0, -self.IL) + self.c_h * max(0, self.IL)) / 200.  # self.config.Ttest #
        self.curReward = -self.curReward
        # make reward negative, because it is the cost

        # sum total reward of each agent
        self.cumReward = self.config.gamma * self.cumReward + self.curReward

    # This function returns a np.array of the current state of the agent
    def getCurState(self, t: int) -> np.ndarray:
        if self.config.ifUseASAO:
            if self.config.if_use_AS_t_plus_1:
                curState = np.array(
                    [-1 * (self.IL < 0) * self.IL, 1 * (self.IL > 0) * self.IL, self.OO, self.AS[t], self.AO[t]]
                )
            else:
                curState = np.array(
                    [-1 * (self.IL < 0) * self.IL, 1 * (self.IL > 0) * self.IL, self.OO, self.AS[t - 1], self.AO[t]]
                )
        else:
            curState = np.array([-1 * (self.IL < 0) * self.IL, 1 * (self.IL > 0) * self.IL, self.OO])

        if self.config.ifUseActionInD:
            a = self.config.actionList[np.argmax(self.action)]
            curState = np.concatenate((curState, np.array([a])))

        return curState
