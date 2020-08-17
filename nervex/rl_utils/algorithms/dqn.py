import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

import numpy as np
import math, random

from collections import namedtuple
from copy import deepcopy

from nervex.utils.log_helper import build_logger, build_logger_naive, pretty_print

from nervex.worker.actor.env_manager.base_env_manager import BaseEnvManager
from nervex.worker.actor.env_manager.vec_env_manager import SubprocessEnvManager
from nervex.envs.gym.pong.pong_env import PongEnv
from nervex.envs.gym.pong.pong_vec_env import PongEnvManager 

from nervex.data.structure.buffer import PrioritizedBuffer


class DqnCNNnetwork(nn.Module):
    def __init__(self, h, w, outputs):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.head(x.view(x.size(0), -1))


class DqnLoss(nn.Module):
    def __init__(self, discount_factor: Optional[float] = 0.99, q_function_criterion=nn.MSELoss()):
        super().__init__()
        self._gamma = discount_factor
        self.q_function_criterion = q_function_criterion

    def forward(self, is_double, q_value, next_q_value, target_q_value, action, reward, terminals, weights=None):
        rewards = reward
        # q_s_a = q_value.gather(1, action.unsqueeze(1).long()).squeeze(1)
        q_s_a = q_value[:, action.unsqueeze(1).long()].squeeze(1)
        target_q_s_a = rewards + self._gamma * (1 - terminals) * \
            target_q_value.gather(1, torch.max(next_q_value, 1)[1].unsqueeze(1)).squeeze(1).to(self.device)
        if weights is not None:
            q_function_loss = self.q_function_criterion(q_s_a, target_q_s_a.detach()) * weights
        else:
            q_function_loss = self.q_function_criterion(q_s_a, target_q_s_a.detach())
        return q_function_loss


class DqnRunner(nn.Module):
    r"""
    Overview:
        The Dqn Algorithnm runner
    Interface:
        __init__, train
    """
    def __init__(
        self,
        cfg,
        q_network,
        # env: Optinal[SubprocessEnvManager] = PongEnvManager,
        env: Optional[BaseEnv] = PongEnv,
        dqn_loss: Optional[DqnLoss] = DqnLoss(),
        opitmizer_type: Optional[str] = 'Adam',
        learning_rate: Optional[float] = 0.001,
        total_frame_num: Optional[int] = 10000,
        batch_size: Optional[int] = 64,
        buffer: Optional[PrioritizedBuffer] = PrioritizedBuffer(1000),
        bandit: Optional[function] = None,
        is_dobule: Optional[bool] = False,
        target_update_freq: Optional[int] = 200,
        device: Optinal[str] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        r"""
        Arguments:
            - q_network,
            - isdouble,
            - usePri,
            - bandit (:obj:`function`)

        """
        self.device = device
        self.cfg = cfg
        self.q_function = q_network.to(self.device)
        #TODO add vec_env, enable multi workers
        # self.env = env
        self.env = env(cfg)
        self.dqn_loss = dqn_loss.to(self.device)
        if opitmizer_type == 'Adam':
            self.opitmizer = optim.Adam(self.q_function.parameters(), learning_rate)
        else:
            self.opitmizer = optim.SGD(self.q_function.parameters(), learning_rate)

        self.batch_size = batch_size
        self.is_dobule = is_dobule

        if is_dobule:
            self.target_q_fuction = deepcopy(self.q_function)
        else:
            self.target_q_fuction = self.q_function
        self.target_update_freq = target_update_freq

        self.total_frame_num = total_frame_num
        self.buffer = buffer.to(self.device)
        if bandit is None:
            self.bandit = lambda x: 0.3
        else:
            self.bandit = bandit

        #TODO
        self.n_actions = 6
        self.logger, self.tb_logger, self.variable_record = build_logger(self.cfg, name="dqn_test")

    def _update_target_networks(self):
        self.target_q_fuction.load_state_dict(self.q_function.state_dict())

    def select_action(self, state, curstep=None):
        sample = random.random()
        if curstep is not None:
            eps_threshold = self.bandit(curstep)
        else:
            eps_threshold = 0.3
        if sample > eps_threshold:
            with torch.no_grad():
                return self.q_function(torch.FloatTensor(state).unsqueeze(0).to(self.device)).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def train(self):
        # print("-------Start Training----------")
        self.logger.info('cfg:\n{}'.format(self.cfg))
        epoch_num = 0
        losses = []
        death = 0
        duration = 0
        self.tb_logger.register_var('loss')
        self.tb_logger.register_var('loss_avg')
        for i_frame in range(self.total_frame_num):
            duration += 1
            print("Start trainging epoch{}".format(epoch_num))
            self.logger.info("=== Training Iteration {} Result ===".format(epoch_num))
            
            state = self.env.reset().transpose((2, 0, 1))
            while True:
                action = self.select_action(state, i_frame)
                next_state, reward, done, _ = self.env.step(action.item())
                next_state = next_state.transpose((2, 0, 1))
                if reward == -1.0:
                    reward = -10.0
                    death += 1
                else:
                    reward = 1.0
                if death >= 5:
                    done = True
                reward = torch.tensor([reward], device=self.device)

                step = {}
                # step['"obs", "acts", "nextobs", "rewards", "termianls"']
                step['obs'] = state
                step['acts'] = action
                step['nextobs'] = next_state
                step['rewards'] = reward
                if done:
                    isdone = torch.ones(1)
                else:
                    isdone = torch.zeros(1)
                step['termianls'] = isdone

                self.buffer.append(step)

                state = next_state

                if self.buffer.validlen < self.batch_size:
                    continue

                batchs = self.buffer.sample(self.batch_size)

                state_batch = torch.cat([torch.Tensor([x['obs']]) for x in batchs], 0).to(self.device)
                nextstate_batch = torch.cat([torch.Tensor([x['nextobs']]) for x in batchs], 0).to(self.device)
                action_batch = torch.cat([torch.IntTensor([x['acts']]) for x in batchs]).to(self.device)
                reward_batch = torch.cat([x['rewards'] for x in batchs]).to(self.device)
                terminate_batch = torch.cat([x['termianls'] for x in batchs]).to(self.device)

                q_value = self.q_function(state_batch.to(self.device))
                next_q_value = self.q_function(nextstate_batch.to(self.device))

                if self.is_dobule:
                    target_q_value = self.target_q_fuction(nextstate_batch.to(self.device))
                else:
                    target_q_value = next_q_value

                loss = self.dqn_loss(q_value, next_q_value, target_q_value, action_batch, reward_batch, terminate_batch)

                self.optimizer.zero_grad()

                self.tb_logger.add_scalar('loss', loss, i_frame)

                losses.append(loss)

                loss.backward()

                self.optimizer.step()

                if i_frame % self.target_update_freq == 0:
                    self._update_target_networks()

                if done:
                    epoch_num += 1
                    if (len(losses) != 0):
                        self.tb_logger.add_scalar('loss_avg', sum(losses) / len(losses), epoch_num)
                    losses = []
                    death = 0
                    duration = 0
                    break


def epsilon_greedy(start, end, decay):
    return lambda x: (start - end) * math.exp(-1 * x / decay) + end


if __name__ == "__main__":
    bandit = epsilon_greedy(0.95, 0.03, 10000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_network = DqnCNNnetwork(210, 160, 6).to(device)
    dqn_runner = DqnRunner(
        cfg={},
        q_network=q_network,
        learning_rate=0.0001,
        total_frame_num=1000000,
        is_dobule=True,
        buffer=PrioritizedBuffer(10000),
        bandit=epsilon_greedy(0.95, 0.05, 50000),
        batch_size=64
    )
    dqn_runner.train()
