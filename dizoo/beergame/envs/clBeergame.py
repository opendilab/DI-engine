# Code Reference: https://github.com/OptMLGroup/DeepBeerInventory-RL.
import numpy as np
from random import randint
from .BGAgent import Agent
from matplotlib import rc
rc('text', usetex=True)
from .plotting import plotting, savePlot
import matplotlib.pyplot as plt
import os
import time
from time import gmtime, strftime


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
    def resetGame(self, demand: np.ndarray):
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
    def distTotReward(self, role: int):
        totRew = 0
        optRew = 0.1  # why?
        for i in range(self.config.NoAgent):
            # sum all rewards for the agents and make correction
            totRew += self.players[i].cumReward
        totRew += optRew

        return totRew, self.players[role].cumReward

    def getAction(self, k: int, action: np.ndarray, playType="train"):
        if playType == "train":
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
        elif playType == "test":
            if self.players[k].compTypeTest == "srdqn":
                self.players[k].action = np.zeros(self.config.actionListLen)
                self.players[k].action = self.players[k].brain.getDNNAction(self.playType)
            elif self.players[k].compTypeTest == "Strm":
                self.players[k].action = np.zeros(self.config.actionListLenOpt)

                self.players[k].action[np.argmin(np.abs(np.array(self.config.actionListOpt)-\
                    max(0,round(self.players[k].AO[self.curTime] +\
                     self.players[k].alpha_b*(self.players[k].IL - self.players[k].a_b) +\
                     self.players[k].betta_b*(self.players[k].OO - self.players[k].b_b)))))] = 1
            elif self.players[k].compTypeTest == "rnd":
                self.players[k].action = np.zeros(self.config.actionListLen)
                a = np.random.randint(self.config.actionListLen)
                self.players[k].action[a] = 1
            elif self.players[k].compTypeTest == "bs":
                self.players[k].action = np.zeros(self.config.actionListLenOpt)

                if self.config.demandDistribution == 2:
                    if self.curTime and self.config.use_initial_BS <= 4:
                        self.players[k].action [np.argmin(np.abs(np.array(self.config.actionListOpt)-\
                          max(0,(self.players[k].int_bslBaseStock - (self.players[k].IL + self.players[k].OO - self.players[k].AO[self.curTime]))) ))] = 1
                    else:
                        self.players[k].action [np.argmin(np.abs(np.array(self.config.actionListOpt)-\
                          max(0,(self.players[k].bsBaseStock - (self.players[k].IL + self.players[k].OO - self.players[k].AO[self.curTime]))) ))] = 1
                else:
                    self.players[k].action [np.argmin(np.abs(np.array(self.config.actionListOpt)-\
                       max(0,(self.players[k].bsBaseStock - (self.players[k].IL + self.players[k].OO - self.players[k].AO[self.curTime]))) ))] = 1
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

    def handelAction(self, action: np.ndarray, playType="train"):
        # get random lead time
        leadTime = randint(self.config.leadRecOrderLow[0], self.config.leadRecOrderUp[0])
        # set AO
        self.players[0].AO[self.curTime] += self.demand[self.curTime]
        for k in range(0, self.config.NoAgent):
            self.getAction(k, action, playType)

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

    def doTestMid(self, demandTs):
        self.resultTest = []
        m = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        self.doTest(m, demandTs)
        print("---------------------------------------------------------------------------------------")
        resultSummary = np.array(self.resultTest).mean(axis=0).tolist()

        result_srdqn = ', '.join(map("{:.2f}".format, resultSummary[0]))
        result_rand = ', '.join(map("{:.2f}".format, resultSummary[1]))
        result_strm = ', '.join(map("{:.2f}".format, resultSummary[2]))
        if self.ifOptimalSolExist:
            result_bs = ', '.join(map("{:.2f}".format, resultSummary[3]))
            print(
                'SUMMARY; {0:s}; ITER= {1:d}; OURPOLICY= [{2:s}]; SUM = {3:2.4f}; Rand= [{4:s}]; SUM = {5:2.4f}; STRM= [{6:s}]; SUM = {7:2.4f}; BS= [{8:s}]; SUM = {9:2.4f}'
                .format(
                    strftime("%Y-%m-%d %H:%M:%S", gmtime()), self.curGame, result_srdqn, sum(resultSummary[0]),
                    result_rand, sum(resultSummary[1]), result_strm, sum(resultSummary[2]), result_bs,
                    sum(resultSummary[3])
                )
            )

        else:
            print(
                'SUMMARY; {0:s}; ITER= {1:d}; OURPOLICY= [{2:s}]; SUM = {3:2.4f}; Rand= [{4:s}]; SUM = {5:2.4f}; STRM= [{6:s}]; SUM = {7:2.4f}'
                .format(
                    strftime("%Y-%m-%d %H:%M:%S", gmtime()), self.curGame, result_srdqn, sum(resultSummary[0]),
                    result_rand, sum(resultSummary[1]), result_strm, sum(resultSummary[2])
                )
            )

        print("=======================================================================================")

    def doTest(self, m, demand):
        import matplotlib.pyplot as plt
        if self.config.ifSaveFigure:
            plt.figure(self.curGame, figsize=(12, 8), dpi=80, facecolor='w', edgecolor='k')

            # self.demand = demand
            # use dnn to get output.
            Rsltdnn, plt = self.tester(self.config.agentTypes, plt, 'b', 'OurPolicy', m)
            baseStockdata = self.players[0].srdqnBaseStock
            # # use random to get output.
            RsltRnd, plt = self.tester(["rnd", "rnd", "rnd", "rnd"], plt, 'y', 'RAND', m)

            # use formual to get output.
            RsltStrm, plt = self.tester(["Strm", "Strm", "Strm", "Strm"], plt, 'g', 'Strm', m)

            # use optimal strategy to get output, if it works.
            if self.ifOptimalSolExist:
                if self.config.agentTypes == ["srdqn", "Strm", "Strm", "Strm"]:
                    Rsltbs, plt = self.tester(["bs", "Strm", "Strm", "Strm"], plt, 'r', 'Strm-BS', m)
                elif self.config.agentTypes == ["Strm", "srdqn", "Strm", "Strm"]:
                    Rsltbs, plt = self.tester(["Strm", "bs", "Strm", "Strm"], plt, 'r', 'Strm-BS', m)
                elif self.config.agentTypes == ["Strm", "Strm", "srdqn", "Strm"]:
                    Rsltbs, plt = self.tester(["Strm", "Strm", "bs", "Strm"], plt, 'r', 'Strm-BS', m)
                elif self.config.agentTypes == ["Strm", "Strm", "Strm", "srdqn"]:
                    Rsltbs, plt = self.tester(["Strm", "Strm", "Strm", "bs"], plt, 'r', 'Strm-BS', m)
                elif self.config.agentTypes == ["srdqn", "rnd", "rnd", "rnd"]:
                    Rsltbs, plt = self.tester(["bs", "rnd", "rnd", "rnd"], plt, 'r', 'RND-BS', m)
                elif self.config.agentTypes == ["rnd", "srdqn", "rnd", "rnd"]:
                    Rsltbs, plt = self.tester(["rnd", "bs", "rnd", "rnd"], plt, 'r', 'RND-BS', m)
                elif self.config.agentTypes == ["rnd", "rnd", "srdqn", "rnd"]:
                    Rsltbs, plt = self.tester(["rnd", "rnd", "bs", "rnd"], plt, 'r', 'RND-BS', m)
                elif self.config.agentTypes == ["rnd", "rnd", "rnd", "srdqn"]:
                    Rsltbs, plt = self.tester(["rnd", "rnd", "rnd", "bs"], plt, 'r', 'RND-BS', m)
                else:
                    Rsltbs, plt = self.tester(["bs", "bs", "bs", "bs"], plt, 'r', 'BS', m)
            # hold the results of the optimal solution
                self.middleTestResult += [[RsltRnd, RsltStrm, Rsltbs]]
            else:
                self.middleTestResult += [[RsltRnd, RsltStrm]]

        else:
            # return the obtained results into their lists
            RsltRnd = self.middleTestResult[m][0]
            RsltStrm = self.middleTestResult[m][1]
            if self.ifOptimalSolExist:
                Rsltbs = self.middleTestResult[m][2]

        # save the figure
        if self.config.ifSaveFigure:
            savePlot(self.players, self.curGame, Rsltdnn, RsltStrm, Rsltbs, RsltRnd, self.config, m)
            plt.close()

        result_srdqn = ', '.join(map("{:.2f}".format, Rsltdnn))
        result_rand = ', '.join(map("{:.2f}".format, RsltRnd))
        result_strm = ', '.join(map("{:.2f}".format, RsltStrm))
        if self.ifOptimalSolExist:
            result_bs = ', '.join(map("{:.2f}".format, Rsltbs))
            print(
                'output; {0:s}; Iter= {1:s}; SRDQN= [{2:s}]; sum = {3:2.4f}; Rand= [{4:s}]; sum = {5:2.4f}; Strm= [{6:s}]; sum = {7:2.4f}; BS= [{8:s}]; sum = {9:2.4f}'
                .format(
                    strftime("%Y-%m-%d %H:%M:%S", gmtime()), str(str(self.curGame) + "-" + str(m)), result_srdqn,
                    sum(Rsltdnn), result_rand, sum(RsltRnd), result_strm, sum(RsltStrm), result_bs, sum(Rsltbs)
                )
            )
            self.resultTest += [[Rsltdnn, RsltRnd, RsltStrm, Rsltbs]]

        else:
            print(
                'output; {0:s}; Iter= {1:s}; SRDQN= [{2:s}]; sum = {3:2.4f}; Rand= [{4:s}]; sum = {5:2.4f}; Strm= [{6:s}]; sum = {7:2.4f}'
                .format(
                    strftime("%Y-%m-%d %H:%M:%S", gmtime()), str(str(self.curGame) + "-" + str(m)), result_srdqn,
                    sum(Rsltdnn), result_rand, sum(RsltRnd), result_strm, sum(RsltStrm)
                )
            )

            self.resultTest += [[Rsltdnn, RsltRnd, RsltStrm]]

        return sum(Rsltdnn)

    def tester(self, testType, plt, colori, labeli, m):

        # set computation type for test
        for k in range(0, self.config.NoAgent):
            # self.players[k].compTypeTest = testType[k]
            self.players[k].compType = testType[k]
        # run the episode to get the results.
        if labeli != 'OurPolicy':
            result = self.playGame(self.demand)
        else:
            result = [-1 * self.players[i].cumReward for i in range(0, self.config.NoAgent)]
        # add the results into the figure
        if self.config.ifSaveFigure:
            plt = plotting(plt, [np.array(self.players[i].hist) for i in range(0, self.config.NoAgent)], colori, labeli)
        if self.config.ifsaveHistInterval and ((self.curGame == 0) or (self.curGame == 1) or (self.curGame == 2) or (self.curGame == 3) or ((self.curGame - 1) % self.config.saveHistInterval == 0)\
         or ((self.curGame) % self.config.saveHistInterval == 0) or ((self.curGame) % self.config.saveHistInterval == 1) \
         or ((self.curGame) % self.config.saveHistInterval == 2)) :
            for k in range(0, self.config.NoAgent):
                name = labeli + "-" + str(self.curGame) + "-" + "player" + "-" + str(k) + "-" + str(m)
                np.save(os.path.join(self.config.model_dir, name), np.array(self.players[k].hist2))

        # save the figure of base stocks
        # if self.config.ifSaveFigure and (self.curGame in range(self.config.saveFigInt[0],self.config.saveFigInt[1])):
        # 	for k in range(self.config.NoAgent):
        # 		if self.players[k].compTypeTest == 'dnn':
        # 			plotBaseStock(self.players[k].srdqnBaseStock, 'b', 'base stock of agent '+ str(self.players[k].agentNum), self.curGame, self.config, m)

        return result, plt

    def playGame(self, demand):
        self.resetGame(demand)

        # run the game
        while self.curTime < self.T:
            self.handelAction(np.array(0))  # action won't be used.
            self.next()
        return [-1 * self.players[i].cumReward for i in range(0, self.config.NoAgent)]
