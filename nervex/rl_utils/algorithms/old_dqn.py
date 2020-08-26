import torch
import numpy as np
from copy import deepcopy
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from typing import Dict, Union, Optional
from nervex.data.structure.buffer import PrioritizedBuffer
import math
import random
from itertools import count
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple

from nervex.envs.env.base_env import BaseEnv
from nervex.envs.gym.pong.pong_env import PongEnv 
from nervex.rl_utils.algorithms.old_dqnloss import DqnLoss

from tensorboardX import SummaryWriter
writer = SummaryWriter("./board_summary")


batch_step = namedtuple("Batch", ["obs", "acts", "nextobs", "rewards", "termianls"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNnetwork(nn.Module):

    def __init__(self, h, w, outputs):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        # self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        # self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        # self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # print(x)
        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.head(x.view(x.size(0), -1))




class DqnRunner(nn.Module):

    def __init__(self,
                # q_network,
                # dqn_loss: nn.Module,
                # optim: torch.optim.Optimizer = optim.Adam(),
                step_num: Optional[int] = 500,
                env: Optional[BaseEnv]= PongEnv,
                discount_factor: Optional[float] = 0.99,
                estimation_step: Optional[int] = 1,
                target_update_freq: Optional[int] = 200,
                q_fucntion_criterion: Optional= nn.MSELoss(),

        ):

######
        q_network = DQNnetwork(210, 160, 6).to(device)
        
        
        super().__init__()
        self.input = input
        # self.optim = optim.Adam
        self.eps = 0.3
        self._gamma = discount_factor
        self.target_update_freq = target_update_freq
        self.step_num = step_num
        self.q_function = q_network
        self.target_q_fuction = deepcopy(self.q_function)
        self.target_q_fuction.load_state_dict(self.q_function.state_dict())
        self.target_q_fuction.eval()
        self.q_fucntion_criterion = q_fucntion_criterion
        
        
        self.env = env({})

        #TODO
        self.n_actions = 6

        self.dqn_loss = DqnLoss(self._gamma, self.q_fucntion_criterion)

        #TODO
        self.batch_size = 128

        #TODO
        self.buffer = PrioritizedBuffer(10000)

        self.optimizer = optim.Adam(self.q_function.parameters())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        #TODO
        self.ucb_max = 0.95
        self.ucb_min = 0.03
        self.ucb_decay = 10000
        #TODO
        # self.eps_ucb_function = lambda curstep: (self.ucb_max - self.ucb_min)*(curstep / self.step_num )+ self.ucb_min
        self.eps_ucb_function = lambda curframe: (self.ucb_max - self.ucb_min)*math.exp(-1 * curframe / self.ucb_decay ) + self.ucb_min

    def set_eps(self, eps):
        self.eps

    def update(self, stepsize):
        for _ in range(stepsize):
            batch = buffer.sample(self.batch_size)
            info = (batch)

    def select_action(self,state, curstep=None):
        # global steps_done
        sample = random.random()
        # eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        #     math.exp(-1. * steps_done / EPS_DECAY)
        # steps_done += 1
        if curstep != None:
            eps_threshold = self.eps_ucb_function(curstep)
        else:
            eps_threshold = self.eps
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                # print(state)
                return self.q_function(torch.FloatTensor(state).unsqueeze(0).to(device)).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def get_state(self):
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. (Transpose it into torch order (CHW).
        screen = self.env.pong_obs.transpose((2, 0, 1))
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        # a = torch.randn(4)
        # print("screen=",screen, a, type(screen), screen.shape, screen.dtype)
        screen = torch.from_numpy(screen)
        # screen = torch.FloatTensor(torch.randn(3, 210,6))
        # print(screen)
        return screen.unsqueeze(0)
        resize = T.Compose([T.ToPILImage(),
                # T.Resize(40, interpolation=Image.CUBIC),
                T.ToTensor()])
        # Resize, and add a batch dimension (BCHW)
        return resize(screen).unsqueeze(0).to(self.device)

    def train(self):
        print("-------Start Training----------")
        total_step = 0
        for i_step in range(self.step_num):
            print("Start trainging epoch{}".format(i_step))
            state = self.env.reset().transpose((2, 0, 1))
            # state = self.get_state()
            # # print("state = ", state)
            # state = self.env.pong_obs.transpose((2, 0, 1))
            losses = []
            death = 0
            duration = 0
            for t in count():
                total_step += 1
                # print("Do epoch{}, action{}".format(i_step, t))
                action = self.select_action(state, total_step)
                next_state, reward, done, _  = self.env.step(action.item())
                next_state = next_state.transpose((2,0,1))
                if reward == -1.0:
                    reward = -10.0
                    death += 1
                else:
                    reward = 1.0
                if death >= 5:
                    done = True
                reward = torch.tensor([reward], device=device)
                # if done:
                #     print("epoch{} done".format(i_step))
                #     # next_state = None
                #     next_state = self.get_state()
                # else:
                #     next_state = self.get_state()
                
                step = {}
                # step['"obs", "acts", "nextobs", "rewards", "termianls"']
                step['obs'] = state
                step['acts']= action
                step['nextobs'] = next_state
                step['rewards'] = reward
                if done:
                    isdone = torch.ones(1)
                else :
                    isdone = torch.zeros(1)
                step['termianls'] = isdone

                #self.buffer.append(batch_step(state, action, next_state, reward, done)._asdict)
                self.buffer.append(step)
                
                # print("buffer vaild length = ", self.buffer.validlen)
                state = next_state

                if self.buffer.validlen < self.batch_size:
                    print("buffer vaildlen too small, continue.")
                    continue

                # print("get batches.")
                batchs = self.buffer.sample(self.batch_size)
                # batch = batch_step(*zip(*batchs))
                # print(batchs)

                # ba = {}


                # batch = batch_step._make([batchs['obs'], batchs['acts'], batchs['nextobs'], batchs['reward'], batchs['termianls']])
                
                # batch = batchs
                # state_batch = torch.cat(batch.obs)
                # action_batch = torch.cat(batch.acts)
                # reward_batch = torch.cat(batch.rewards)
                # terminate_batch = torch.cat(batch.termianls)

                state_batch = torch.cat([torch.Tensor([x['obs']]) for x in batchs], 0).to(device)
                nextstate_batch = torch.cat([torch.Tensor([x['nextobs']]) for x in batchs], 0).to(device)
                action_batch = torch.cat([torch.IntTensor([x['acts']]) for x in batchs]).to(device)
                reward_batch = torch.cat([x['rewards'] for x in batchs]).to(device)
                terminate_batch = torch.cat([x['termianls'] for x in batchs]).to(device)

                # state_batch = torch.cat([x['obs'] for x in batchs]).to(device)
                # nextstate_batch = torch.cat([x['obs'] for x in batchs]).to(device)
                # action_batch = torch.cat([x['acts'] for x in batchs]).to(device)
                # reward_batch = torch.cat([x['rewards'] for x in batchs]).to(device)
                # terminate_batch = torch.cat([x['termianls'] for x in batchs]).to(device)

                q_value = self.q_function(state_batch.to(device))

                # print("q_value is =", q_value)

                # print("actions are=", action_batch)

                next_q_value = self.q_function(nextstate_batch.to(device))

                # print("next q_value is =", next_q_value)

                target_q_value = self.target_q_fuction(nextstate_batch.to(device))

                # print("target_q_value is = ", target_q_value)

                loss = self.dqn_loss(q_value, next_q_value, target_q_value, action_batch, reward_batch, terminate_batch)

                self.optimizer.zero_grad()
                
                loss.backward()

                self.optimizer.step()

                if total_step % self.target_update_freq == 0:
                    self._update_target_networks()
                # for param in self.q_function.parameters():
                #     param.grad.data.clamp_(-1, 1)
                # self.optimizer.step()
                if done:
                    break
                duration = t
            if(len(losses) != 0):
                writer.add_scalar("training_loss", sum(losses)/len(losses), i_step)
                print("epoch:", i_step, ", duration==", duration, " , average loss==", sum(losses)/len(losses), ", lenloss ==", len(losses))

            writer.add_scalar("duration", len(losses), i_step)
            writer.add_scalar("t", duration, i_step)
            writer.flush()
            
            

    def _update_target_networks(self):
        self.target_q_fuction.load_state_dict(self.q_function.state_dict())
        

# q_function = DQNnetwork(210, 160, 6).to(device)


# dqn_runner = DqnRunner(q_function)

dqn_runner = DqnRunner()

dqn_runner.train()

writer.close()