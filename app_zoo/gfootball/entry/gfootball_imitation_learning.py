import math
import threading
import time

import numpy as np
import torch
import torch.nn as nn

from app_zoo.gfootball.model.iql.iql_network import FootballIQL
from app_zoo.gfootball.model.rule_based_bot import FootballRuleBaseModel
from nervex.torch_utils import Adam
from nervex.torch_utils.network.nn_module import one_hot
from nervex.torch_utils import to_tensor, to_ndarray, to_list
from nervex.data import default_collate
from nervex.data.structure import ReplayBuffer

try:
    from app_zoo.gfootball.envs.gfootball_env import GfootballEnv
except ModuleNotFoundError:
    print("[WARNING] no gfootball env, if you want to use gfootball, please install it, otherwise, ignore it.")

import gfootball.env as football_env


class GfootballIL(object):

    def __init__(self, cfg):
        self._env = GfootballEnv({})
        self._model = FootballIQL({})
        self._bot = FootballRuleBaseModel()
        self._optimizer = Adam(self._model.parameters(), weight_decay=0.0001)
        self._replay_buffer = ReplayBuffer(
            'agent',
            load_path=None,
            maxlen=100000,
            max_reuse=10,
            max_staleness=10000,
            min_sample_ratio=1.5,
            alpha=0.6,
            # (Float type)  How much correction is used: 0 means no correction while 1 means full correction
            beta=0.4,
            # Anneal step for beta: 0 means no annealing
            anneal_step=0,
            # Whether to track the used data or not. Buffer will use a new data structure to track data if set True.
            enable_track_used_data=False,
            # Whether to deepcopy data when willing to insert and sample data. For security purpose.
            deepcopy=False,
        )
        self._learner_step = 0
        self._gamma = cfg.get('gamma', 1.0)
        self._batch_size = cfg.get('batch_size', 16)
        self._ckpt_freq = cfg.get("ckpt_freq", 1000)
        self._eval_freq = cfg.get("eval_freq", 1000)
        self._eval_game_num = cfg.get("eval_game_num", 5)
        self._loss_print_freq = cfg.get("loss_freq", 100)
        self._max_learner_step = cfg.get("max_learner_step", 100000)
        self._max_bot_episode = cfg.get("max_bot_episode", 5000)

    def get_episode_bot_data(self, episode_num):
        episode = 0
        datas = []
        while episode < episode_num:
            episode += 1
            obs = self._env.reset()
            steps = 0
            observations = []
            rewards = []
            actions = []
            while True:
                bot_action = self._bot([obs['raw_obs']])
                obs, rew, done, info = self._env.step(bot_action)
                rewards.append(rew)
                observations.append(obs)
                actions.append(bot_action)
                steps += 1
                if done:
                    pre_rew = 0
                    for i in range(len(rewards) - 1, -1, -1):
                        data = {}
                        data['obs'] = observations[i]
                        data['action'] = actions[i]
                        cur_rew = rewards[i]
                        pre_rew = cur_rew + (pre_rew * self._gamma)
                        # TODO find a better function here
                        data['priority'] = math.e ** float(pre_rew)
                        datas.append(data)
                    break
        return datas

    def _train(self):
        data = self._replay_buffer.sample(self._batch_size, self._learner_step)
        self._learner_step += 1

        obs = [d['obs']['processed_obs'] for d in data]
        action = torch.stack([one_hot(d['action'], 19)[0] for d in data])

        model_action = self._model(default_collate(obs, cat_1dim=False))

        loss = nn.MSELoss(reduction='none')(model_action, action).mean()
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        # print("train iter{} loss: {}".format(self._learner_step, loss.item()))

        if self._learner_step % self._ckpt_freq == 0:
            torch.save(self._model, "model_learner_iter{}.pt".format(self._learner_step))

        return loss

    def train(self):
        losses = []
        while self._learner_step < self._max_learner_step:
            loss = self._train()
            losses.append(loss.item())
            if self._learner_step % self._loss_print_freq == 0:
                print(
                    "Step {} to {} mean loss: {}".format(
                        self._learner_step - self._loss_print_freq, self._learner_step,
                        sum(losses) / len(losses) + 1e-6
                    )
                )
            if self._learner_step % self._eval_freq == 0:
                print("Learner step {}, start eval current model".format(self._learner_step))
                self.eval_current_model(self._eval_game_num)

    def generate_bot_data(self):
        while True:
            self._replay_buffer.extend(self.get_episode_bot_data(1))

    def run(self):
        losses = []
        while self._learner_step < self._max_learner_step:
            self._replay_buffer.extend(self.get_episode_bot_data(1))
            for i in range(20):
                loss = self._train()
                losses.append(loss.item())
                if self._learner_step % self._loss_print_freq == 0:
                    print(
                        "Step {} to {} mean loss: {}".format(
                            self._learner_step - self._loss_print_freq, self._learner_step,
                            sum(losses) / len(losses) + 1e-6
                        )
                    )
                if self._learner_step % self._eval_freq == 0:
                    print("Learner step {}, start eval current model".format(self._learner_step))
                    self.eval_current_model(self._eval_game_num)

    def run_with_threads(self):
        data_thread = threading.Thread(target=self.generate_bot_data)
        learn_thread = threading.Thread(target=self.train)
        data_thread.start()
        time.sleep(120)
        learn_thread.start()

    def eval_current_model(self, eval_episode):
        episode = 0
        while episode < eval_episode:
            episode += 1
            print("Start Episode #", episode, " game: ")
            obs = self._env.reset()
            steps = 0
            while True:
                model_action = self._model(default_collate([obs['processed_obs']], cat_1dim=False))
                obs, rew, done, info = self._env.step(model_action[0].max(0)[1])
                steps += 1
                if done:
                    print("Episode #", episode, "end with reward:", info)
                    break


if __name__ == "__main__":
    gfootballIL = GfootballIL({})
    gfootballIL.run()
