import numpy as np
from random import randint
from .BGAgent import Agent
from matplotlib import rc
rc('text', usetex=True)


class clBeerGame(object):

    def __init__(self, config):
        self.config = config
        self.curGame = 0  # The number associated with the current game (counter of the game)
        self.curTime = 0
        self.totIterPlayed = 0  # total iterations of the game, played so far in this and previous games
        self.players = self.createAgent()  # create the agents
        self.T = 0
        self.demand = []
        self.ifOptimalSolExist = self.config.ifOptimalSolExist
        self.getOptimalSol()
        self.totRew = 0  # it is reward of all players obtained for the current player.
        self.resultTest = []
        self.runnerMidlResults = []  # stores the results to use in runner comparisons
        self.runnerFinlResults = []  # stores the results to use in runner comparisons
        self.middleTestResult = [
        ]  # stores the whole middle results of bs, Strm, and random to avoid doing same tests multiple of times.
        self.runNumber = 0  # the runNumber which is used when use runner
        self.strNum = 0  # the runNumber which is used when use runner

    # createAgent : Create agent objects (agentNum,IL,OO,c_h,c_p,type,config)
    def createAgent(self):
        agentTypes = self.config.agentTypes
        return [
            Agent(
                i, self.config.ILInit[i], self.config.AOInit, self.config.ASInit[i], self.config.c_h[i],
                self.config.c_p[i], self.config.eta[i], agentTypes[i], self.config
            ) for i in range(self.config.NoAgent)
        ]

    # planHorizon : Find a random planning horizon
    def planHorizon(self):
        # TLow: minimum number for the planning horizon # TUp: maximum number for the planning horizon
        # output: The planning horizon which is chosen randomly.
        return randint(self.config.TLow, self.config.TUp)

    # this function resets the game for start of the new game
    def resetGame(self, demand):
        self.demand = demand
        self.curTime = 0
        self.curGame += 1
        self.totIterPlayed += self.T
        self.T = self.planHorizon()
        # reset the required information of player for each episode
        for k in range(0, self.config.NoAgent):
            self.players[k].resetPlayer(self.T)

        # update OO when there are initial IL,AO,AS
        self.update_OO()

    # correction on cost at time T according to the cost of the other players
    def getTotRew(self):
        totRew = 0
        for i in range(self.config.NoAgent):
            # sum all rewards for the agents and make correction
            totRew += self.players[i].cumReward

        for i in range(self.config.NoAgent):
            self.players[i].curReward += self.players[i].eta * (totRew - self.players[i].cumReward)  # /(self.T)

    # make correction to the rewards in the experience replay for all iterations of current game
    def distTotReward(self, role):
        totRew = 0
        optRew = 0.1  # why?
        for i in range(self.config.NoAgent):
            # sum all rewards for the agents and make correction
            totRew += self.players[i].cumReward
        totRew += optRew

        return totRew, self.players[role].cumReward

    def getAction(self, k, action):
        if self.players[k].compType == "srdqn":
            self.players[k].action = np.zeros(self.config.actionListLen)
            self.players[k].action[action] = 1
        elif self.players[k].compType == "Strm":
            self.players[k].action = np.zeros(self.config.actionListLenOpt)
            self.players[k].action[np.argmin(np.abs(np.array(self.config.actionListOpt)\
                  - max(0, round(self.players[k].AO[self.curTime] + \
                 self.players[k].alpha_b*(self.players[k].IL - self.players[k].a_b) + \
                 self.players[k].betta_b*(self.players[k].OO - self.players[k].b_b)))))] = 1
        elif self.players[k].compType == "rnd":
            self.players[k].action = np.zeros(self.config.actionListLen)
            a = np.random.randint(self.config.actionListLen)
            self.players[k].action[a] = 1
        elif self.players[k].compType == "bs":
            self.players[k].action = np.zeros(self.config.actionListLenOpt)
            if self.config.demandDistribution == 2:
                if self.curTime and self.config.use_initial_BS <= 4:
                    self.players[k].action[np.argmin(np.abs(np.array(self.config.actionListOpt) - \
                      max(0, (self.players[k].int_bslBaseStock - (self.players[k].IL + self.players[k].OO - self.players[k].AO[self.curTime])))))] = 1
                else:
                    self.players[k].action[np.argmin(np.abs(np.array(self.config.actionListOpt) - \
                      max(0, (self.players[k].bsBaseStock - (self.players[k].IL + self.players[k].OO - self.players[k].AO[self.curTime])))))] = 1
            else:
                self.players[k].action[np.argmin(np.abs(np.array(self.config.actionListOpt) - \
                   max(0, (self.players[k].bsBaseStock - (self.players[k].IL + self.players[k].OO - self.players[k].AO[self.curTime])))))] = 1
        else:
            # not a valid player is defined.
            raise Exception('The player type is not defined or it is not a valid type.!')

    def next(self):
        # get a random leadtime
        leadTimeIn = randint(
            self.config.leadRecItemLow[self.config.NoAgent - 1], self.config.leadRecItemUp[self.config.NoAgent - 1]
        )
        # handle the most upstream recieved shipment
        self.players[self.config.NoAgent - 1].AS[self.curTime +
                                                 leadTimeIn] += self.players[self.config.NoAgent -
                                                                             1].actionValue(self.curTime)

        for k in range(self.config.NoAgent - 1, -1, -1):  # [3,2,1,0]

            # get current IL and Backorder
            current_IL = max(0, self.players[k].IL)
            current_backorder = max(0, -self.players[k].IL)

            # TODO: We have get the AS and AO from the UI and update our AS and AO, so that code update the corresponding variables

            # increase IL and decrease OO based on the action, for the next period
            self.players[k].recieveItems(self.curTime)

            # observe the reward
            possible_shipment = min(
                current_IL + self.players[k].AS[self.curTime], current_backorder + self.players[k].AO[self.curTime]
            )

            # plan arrivals of the items to the downstream agent
            if self.players[k].agentNum > 0:
                leadTimeIn = randint(self.config.leadRecItemLow[k - 1], self.config.leadRecItemUp[k - 1])
                self.players[k - 1].AS[self.curTime + leadTimeIn] += possible_shipment

            # update IL
            self.players[k].IL -= self.players[k].AO[self.curTime]
            # observe the reward
            self.players[k].getReward()
            self.players[k].hist[-1][-2] = self.players[k].curReward
            self.players[k].hist2[-1][-2] = self.players[k].curReward

            # update next observation
            self.players[k].nextObservation = self.players[k].getCurState(self.curTime + 1)

        if self.config.ifUseTotalReward:
            # correction on cost at time T
            if self.curTime == self.T:
                self.getTotRew()

        self.curTime += 1

    def handelAction(self, action):
        # get random lead time
        leadTime = randint(self.config.leadRecOrderLow[0], self.config.leadRecOrderUp[0])
        # set AO
        self.players[0].AO[self.curTime] += self.demand[self.curTime]
        for k in range(0, self.config.NoAgent):
            self.getAction(k, action)

            self.players[k].srdqnBaseStock += [self.players[k].actionValue( \
             self.curTime) + self.players[k].IL + self.players[k].OO]

            # update hist for the plots
            self.players[k].hist += [[self.curTime, self.players[k].IL, self.players[k].OO,\
               self.players[k].actionValue(self.curTime), self.players[k].curReward, self.players[k].srdqnBaseStock[-1]]]

            if self.players[k].compType == "srdqn":
                self.players[k].hist2 += [[self.curTime, self.players[k].IL, self.players[k].OO, self.players[k].AO[self.curTime], self.players[k].AS[self.curTime], \
                  self.players[k].actionValue(self.curTime), self.players[k].curReward, \
                  self.config.actionList[np.argmax(self.players[k].action)]]]

            else:
                self.players[k].hist2 += [[self.curTime, self.players[k].IL, self.players[k].OO, self.players[k].AO[self.curTime], self.players[k].AS[self.curTime], \
                  self.players[k].actionValue(self.curTime), self.players[k].curReward, 0]]

            # updates OO and AO at time t+1
            self.players[k].OO += self.players[k].actionValue(self.curTime)  # open order level update
            leadTime = randint(self.config.leadRecOrderLow[k], self.config.leadRecOrderUp[k])
            if self.players[k].agentNum < self.config.NoAgent - 1:
                self.players[k + 1].AO[self.curTime + leadTime] += self.players[k].actionValue(
                    self.curTime
                )  # open order level update

    # check the Shang and Song (2003) condition, and if it works, obtains the base stock policy values for each agent
    def getOptimalSol(self):
        # if self.config.NoAgent !=1:
        if self.config.NoAgent != 1 and 1 == 2:
            # check the Shang and Song (2003) condition.
            for k in range(self.config.NoAgent - 1):
                if not (self.players[k].c_h == self.players[k + 1].c_h and self.players[k + 1].c_p == 0):
                    self.ifOptimalSolExist = False

            # if the Shang and Song (2003) condition satisfied, it runs the algorithm
            if self.ifOptimalSolExist == True:
                calculations = np.zeros((7, self.config.NoAgent))
                for k in range(self.config.NoAgent):
                    # DL_high
                    calculations[0][k] = ((self.config.leadRecItemLow + self.config.leadRecItemUp + 2) / 2 \
                           + (self.config.leadRecOrderLow + self.config.leadRecOrderUp + 2) / 2) * \
                         (self.config.demandUp - self.config.demandLow - 1)
                    if k > 0:
                        calculations[0][k] += calculations[0][k - 1]
                    # probability_high
                    nominator_ch = 0
                    low_denominator_ch = 0
                    for j in range(k, self.config.NoAgent):
                        if j < self.config.NoAgent - 1:
                            nominator_ch += self.players[j + 1].c_h
                        low_denominator_ch += self.players[j].c_h
                    if k == 0:
                        high_denominator_ch = low_denominator_ch
                    calculations[2][k] = (self.players[0].c_p +
                                          nominator_ch) / (self.players[0].c_p + low_denominator_ch + 0.0)
                    # probability_low
                    calculations[3][k] = (self.players[0].c_p +
                                          nominator_ch) / (self.players[0].c_p + high_denominator_ch + 0.0)
                # S_high
                calculations[4] = np.round(np.multiply(calculations[0], calculations[2]))
                # S_low
                calculations[5] = np.round(np.multiply(calculations[0], calculations[3]))
                # S_avg
                calculations[6] = np.round(np.mean(calculations[4:6], axis=0))
                # S', set the base stock values into each agent.
                for k in range(self.config.NoAgent):
                    if k == 0:
                        self.players[k].bsBaseStock = calculations[6][k]

                    else:
                        self.players[k].bsBaseStock = calculations[6][k] - calculations[6][k - 1]
                        if self.players[k].bsBaseStock < 0:
                            self.players[k].bsBaseStock = 0
        elif self.config.NoAgent == 1:
            if self.config.demandDistribution == 0:
                self.players[0].bsBaseStock = np.ceil(
                    self.config.c_h[0] / (self.config.c_h[0] + self.config.c_p[0] + 0.0)
                ) * ((self.config.demandUp - self.config.demandLow - 1) / 2) * self.config.leadRecItemUp
        elif 1 == 1:
            f = self.config.f
            f_init = self.config.f_init
            for k in range(self.config.NoAgent):
                self.players[k].bsBaseStock = f[k]
                self.players[k].int_bslBaseStock = f_init[k]

    def update_OO(self):
        for k in range(0, self.config.NoAgent):
            if k < self.config.NoAgent - 1:
                self.players[k].OO = sum(self.players[k + 1].AO) + sum(self.players[k].AS)
            else:
                self.players[k].OO = sum(self.players[k].AS)
