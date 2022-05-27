from collections import namedtuple
import numpy as np
import gym

from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
import copy

P = [1e-3, 1.5e-3, 2e-3, 5e-2, 0.5, 1]
packet_arrive_lambda_all = [30,50,100]
packet_length_mu_all = [10]
packet_length_sigma_all = [2]

maximun_step = 10000

Codebooks_Num = 50
IoT_users = 220

action = [i%Codebooks_Num for i in range(IoT_users)]

#@ENV_REGISTRY.register('scma')
class SCMAEnv():

    def __init__(self):
        self.success = np.zeros(IoT_users)
        self.send = np.zeros(IoT_users)
        self.time_interval = np.zeros(IoT_users)
        self.resend = np.zeros(IoT_users)
        self.codebooks = np.zeros(Codebooks_Num)
        self.packet_length = np.zeros(IoT_users)
        self.origin_packet_length = np.zeros(IoT_users)
        self.final = np.zeros(IoT_users)
        self.codebooks = np.zeros(Codebooks_Num)
        self.now_step = 0
        self.final_eval_reward = 0
        self.n_agents = IoT_users

    def reset(self):
        self.codebooks = np.zeros(Codebooks_Num)
        self.success = np.zeros(IoT_users)
        self.send = np.zeros(IoT_users)
        self.time_interval = np.zeros(IoT_users)
        self.resend = np.zeros(IoT_users)
        self.codebooks = np.zeros(Codebooks_Num)
        self.packet_length = np.zeros(IoT_users)
        self.origin_packet_length = np.zeros(IoT_users)
        self.final = np.zeros(IoT_users)
        self.now_step = 0
        self.final_eval_reward = 0
        self.n_agents = IoT_users
        agent_obs = self.get_obs()
        global_obs = self.get_specific_state()

        return {
            'agent_state': self.get_obs(),
            'global_state': self.get_specific_state(),
        }

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def get_obs(self):
        obs_n = []
        for a in range(self.n_agents):
            obs_n.append(self.get_obs_agent(a))
        return np.array(obs_n).astype(np.float32)

    def get_obs_agent(self, agent_id):
        if agent_id<int(Codebooks_Num*1/2):
            packet_arrive_lambda = packet_arrive_lambda_all[0]
        elif agent_id<int(Codebooks_Num*3/4):
            packet_arrive_lambda = packet_arrive_lambda_all[1]
        else:
            packet_arrive_lambda = packet_arrive_lambda_all[2]
        packet_arrive_lambda = packet_arrive_lambda_all[(agent_id%len(packet_arrive_lambda_all))]
        packet_length_mu = packet_length_mu_all[0]
        packet_length_sigma = packet_length_sigma_all[0]
        agent_obs = np.append(self.codebooks, [packet_arrive_lambda, packet_length_mu, packet_length_sigma])
        agent_obs = np.append(agent_obs, [self.send[agent_id], self.resend[agent_id], agent_id])
        return np.array(agent_obs).astype(np.float32)

    def send_package(self, agent_id):
        final = False
        #if agent_id<int(Codebooks_Num*3/4):
        #    packet_arrive_lambda = packet_arrive_lambda_all[0]
        #elif agent_id<int(Codebooks_Num*4/5):
        #    packet_arrive_lambda = packet_arrive_lambda_all[1]
        #else:
        #    packet_arrive_lambda = packet_arrive_lambda_all[2]

        packet_arrive_lambda = packet_arrive_lambda_all[(agent_id%len(packet_arrive_lambda_all))]
        packet_length_mu = packet_length_mu_all[0]
        packet_length_sigma = packet_length_sigma_all[0]

        if self.resend[agent_id]:  #一旦碰了所有的都重发
            self.send[agent_id] = True
            self.packet_length[agent_id] = self.origin_packet_length[agent_id]
            self.resend[agent_id] = 0
        if self.send[agent_id] == False and self.time_interval[agent_id] == 0:
            self.time_interval[agent_id] = abs(round(np.random.exponential(packet_arrive_lambda)))
        elif self.send[agent_id] == False and self.time_interval[agent_id] > 0:
            self.time_interval[agent_id] = self.time_interval[agent_id] - 1
            if (self.time_interval[agent_id] <= 0):
                self.send[agent_id] = True
                self.packet_length[agent_id] = abs(round(np.random.normal(packet_length_mu, packet_length_sigma)))
                self.origin_packet_length[agent_id] = self.packet_length[agent_id]
        elif self.send[agent_id] == True and self.packet_length[agent_id] > 0:
            self.packet_length[agent_id] = self.packet_length[agent_id] - 1
            if (self.packet_length[agent_id] <= 0):
                self.send[agent_id] = False
                self.time_interval[agent_id] = 0
                self.final[agent_id] = True

    def get_state(self):
        global_obs = np.append(self.codebooks, self.send)
        global_obs = np.append(global_obs, self.resend)
        global_obs = np.append(global_obs, self.success)

        return np.array(global_obs).astype(np.float32)

    def get_agent_specific_state(self, agent_id):
        if agent_id<int(Codebooks_Num*1/2):
            packet_arrive_lambda = packet_arrive_lambda_all[0]
        elif agent_id<int(Codebooks_Num*3/4):
            packet_arrive_lambda = packet_arrive_lambda_all[1]
        else:
            packet_arrive_lambda = packet_arrive_lambda_all[2]
        #packet_arrive_lambda = packet_arrive_lambda_all[(agent_id%len(packet_arrive_lambda_all))]
        packet_length_mu = packet_length_mu_all[0]
        packet_length_sigma = packet_length_sigma_all[0]
        global_obs = np.append(self.codebooks, self.send)
        global_obs = np.append(global_obs, self.resend)
        global_obs = np.append(global_obs, self.success)
        global_obs = np.append(global_obs, [packet_arrive_lambda, packet_length_mu, packet_length_sigma, agent_id])

        return np.array(global_obs).astype(np.float32)

    def get_specific_state(self):
        obs_n = []
        for a in range(self.n_agents):
            obs_n.append(self.get_agent_specific_state(a))
        return np.array(obs_n).astype(np.float32)

    def receive_package(self, action):
        for agent_id in range(IoT_users):
            if self.send[agent_id]:
                self.codebooks[action[agent_id]] = self.codebooks[action[agent_id]] + 1
        #print(self.codebooks)

    def detect_receive_results(self, action, agent_id):
        if self.send[agent_id]:
            if self.codebooks[action[agent_id]]<len(P):
                k = P[int(self.codebooks[action[agent_id]])]
            else:
                k=1
            #print(self.codebooks[action[agent_id]])
            #print(k)
            prob = np.array([k, 1 - k])
            result = np.random.choice([0, 1], p=prob.ravel())
            if result:
                if self.final[agent_id]:
                    self.final[agent_id] = 0
                    self.success[agent_id] = 1
            else:
                self.resend[agent_id] = 1
            #print(self.codebooks)

    def after_receive(self, action, agent_id):
        if self.send[agent_id]:
            self.codebooks[action[agent_id]] = self.codebooks[action[agent_id]] - 1

    def calculate_reward(self):
        success_reward = self.success.sum()
        self.success = np.zeros(IoT_users)
        failed_reward = self.resend.sum()
        final_reward = success_reward - failed_reward
        return np.array(final_reward).astype(np.float32)

    def step(self, action: np.ndarray):
        done = False
        for agent_id in range(self.n_agents):
            self.send_package(agent_id)
        self.receive_package(action)
        for agent_id in range(self.n_agents):
            self.detect_receive_results(action, agent_id)
        for agent_id in range(self.n_agents):
            self.after_receive(action, agent_id)
        agent_obs = self.get_obs()
        global_obs = self.get_specific_state()
        #print(global_obs.shape)
        reward = self.calculate_reward()
        self.final_eval_reward += reward
        self.now_step = self.now_step + 1
        info = {}
        if self.now_step > maximun_step:
            done = True
            info['final_eval_reward'] = self.final_eval_reward
        obs = {
            'agent_state': agent_obs,
            'global_state': global_obs,
        }
        return BaseEnvTimestep(obs, reward, done, info)

    def close(self):
        self.reset()




    def __repr__(self):
        return 'SCMA Codebook environment'


env = SCMAEnv()
#env.seed(0,False)
env.reset()
final_rew =0
file_name = 'collipse_recover.txt'

p0 = np.ones(Codebooks_Num)
p1 = copy.deepcopy(p0)
for i in range(int(Codebooks_Num*10/20)):
    p1[i] = p1[i] + 3/100

p2 = copy.deepcopy(p0)
for i in range(int(Codebooks_Num*16/20),Codebooks_Num):
    p2[i] = p2[i] + 3/100
    p0[i] = p0[i] - 3/100

temp0 = sum(p0)
temp1 = sum(p1)
temp2 = sum(p2)

for i in range(len(p0)):
    p0[i] = p0[i]/temp0
    p1[i] = p1[i]/temp1
    p2[i] = p2[i]/temp2

action = np.zeros(IoT_users)

for i in range(10000):

    for i in range(IoT_users):
        if i%len(packet_arrive_lambda_all)==0:
            action[i] = np.random.choice([i for i in range(Codebooks_Num)], p=p0.ravel())
        elif i%len(packet_arrive_lambda_all)==1:
            action[i] = np.random.choice([i for i in range(Codebooks_Num)], p=p1.ravel())
        elif i%len(packet_arrive_lambda_all)==2:
            action[i] = np.random.choice([i for i in range(Codebooks_Num)], p=p2.ravel())
    action = action.astype(int)
    #print(action)
    #action = np.random.randint(low=0,high=Codebooks_Num,size=(IoT_users),dtype='int')
    _,rew,_,_ = env.step(action)
    #print(action,rew)
    #print(env.send)
    #print(env.resend)
    #print(env.packet_length)

    #print(env.codebooks)
    final_rew = final_rew+rew
    #print(rew)
print(final_rew)
