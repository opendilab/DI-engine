from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import struct
import random
import math
from copy import deepcopy
import queue
from threading import Thread
from collections import deque
import io
import zmq

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from gym.spaces import prng
from gym.spaces.discrete import Discrete
from gym import spaces

from sc2learner.agents.replay_memory import Transition
from sc2learner.agents.replay_memory import RemoteReplayMemory
from sc2learner.utils.utils import tprint


class DQNAgent(object):

  def __init__(self, network, action_space, init_model_path=None):
    assert type(action_space) == spaces.Discrete
    self._action_space = action_space
    self._network = network
    if init_model_path is not None:
      self.load_params(torch.load(init_model_path,
                                  map_location=lambda storage, loc: storage))
    if torch.cuda.device_count() > 1:
      self._network = nn.DataParallel(self._network)
    if torch.cuda.is_available(): self._network.cuda()
    self._optimizer = None
    self._target_network = None
    self._num_optim_steps = 0

  def act(self, observation, eps=0):
    self._network.eval()
    if random.uniform(0, 1) >= eps:
      observation = torch.from_numpy(np.expand_dims(observation, 0))
      if torch.cuda.is_available():
        observation = observation.pin_memory().cuda(non_blocking=True)
      with torch.no_grad():
        q = self._network(observation)
        action = q.data.max(1)[1].item()
      return action
    else:
      return self._action_space.sample()

  def optimize_step(self,
                    obs_batch,
                    next_obs_batch,
                    action_batch,
                    reward_batch,
                    done_batch,
                    mc_return_batch,
                    discount,
                    mmc_beta,
                    gradient_clipping,
                    adam_eps,
                    learning_rate,
                    target_update_interval):
    # create optimizer
    if self._optimizer is None:
      self._optimizer = optim.Adam(self._network.parameters(),
                                   eps=adam_eps,
                                   lr=learning_rate)
    # create target network
    if self._target_network is None:
      self._target_network = deepcopy(self._network)
      if torch.cuda.is_available(): self._target_network.cuda()
      self._target_network.eval()

    # update target network
    if self._num_optim_steps % target_update_interval == 0:
      self._target_network.load_state_dict(self._network.state_dict())

    # move to gpu
    if torch.cuda.is_available():
      obs_batch = obs_batch.cuda(non_blocking=True)
      next_obs_batch = next_obs_batch.cuda(non_blocking=True)
      action_batch = action_batch.cuda(non_blocking=True)
      reward_batch = reward_batch.cuda(non_blocking=True)
      mc_return_batch = mc_return_batch.cuda(non_blocking=True)
      done_batch = done_batch.cuda(non_blocking=True)

    # compute max-q target
    self._network.eval()
    with torch.no_grad():
      q_next_target = self._target_network(next_obs_batch)
      q_next = self._network(next_obs_batch)
      futures = q_next_target.gather(
          1, q_next.max(dim=1)[1].view(-1, 1)).squeeze()
      futures = futures * (1 - done_batch)
      target_q = reward_batch + discount * futures
      target_q = target_q * mmc_beta + (1.0 - mmc_beta) * mc_return_batch

    # define loss
    self._network.train()
    q = self._network(obs_batch).gather(1, action_batch.view(-1, 1)).squeeze()
    loss = F.mse_loss(q, target_q.detach())

    # compute gradient and update parameters
    self._optimizer.zero_grad()
    loss.backward()
    for param in self._network.parameters():
      param.grad.data.clamp_(-gradient_clipping, gradient_clipping)
    self._optimizer.step()
    self._num_optim_steps += 1
    return loss.data.item()

  def reset(self):
    pass

  def load_params(self, state_dict):
    self._network.load_state_dict(state_dict)

  def read_params(self):
    if torch.cuda.device_count() > 1:
      return self._network.module.state_dict()
    else:
      return self._network.state_dict()


class DQNActor(object):

  def __init__(self,
               memory_size,
               memory_warmup_size,
               env,
               network,
               discount,
               send_freq=4.0,
               ports=("5700", "5701", "5702"),
               learner_ip="localhost"):
    assert type(env.action_space) == spaces.Discrete
    assert len(ports) == 3
    self._env = env
    self._discount = discount
    self._epsilon = 1.0

    self._agent = DQNAgent(network, env.action_space)
    self._replay_memory = RemoteReplayMemory(
        is_server=False,
        memory_size=memory_size,
        memory_warmup_size=memory_warmup_size,
        send_freq=send_freq,
        ports=ports[:2],
        server_ip=learner_ip)

    self._zmq_context = zmq.Context()
    self._model_requestor = self._zmq_context.socket(zmq.REQ)
    self._model_requestor.connect("tcp://%s:%s" % (learner_ip, ports[2]))

  def run(self):
    while True:
      # fetch model
      t = time.time()
      self._update_model()
      tprint("Update model time: %f eps: %f" % (time.time() - t, self._epsilon))
      # rollout
      t = time.time()
      self._rollout()
      tprint("Rollout time: %f" % (time.time() - t))

  def _rollout(self):
    rollout, done = [], False
    observation = self._env.reset()
    while not done:
      action = self._agent.act(observation, eps=self._epsilon)
      next_observation, reward, done, info = self._env.step(action)
      rollout.append(
          (observation, action, reward, next_observation, done))
      observation = next_observation

    discounted_return = 0
    for transition in reversed(rollout):
      reward = transition[2]
      discounted_return = discounted_return * self._discount + reward
      self._replay_memory.push(*transition, discounted_return)

  def _update_model(self):
      self._model_requestor.send_string("request model")
      file_object = io.BytesIO(self._model_requestor.recv_pyobj())
      self._agent.load_params(
          torch.load(file_object, map_location=lambda storage, loc: storage))
      self._epsilon = self._model_requestor.recv_pyobj()


class DQNLearner(object):

  def __init__(self,
               network,
               action_space,
               memory_size,
               memory_warmup_size,
               discount,
               eps_start,
               eps_end,
               eps_decay_steps,
               eps_decay_steps2,
               batch_size,
               mmc_beta,
               gradient_clipping,
               adam_eps,
               learning_rate,
               target_update_interval,
               checkpoint_dir,
               checkpoint_interval,
               print_interval,
               ports=("5700", "5701", "5702"),
               init_model_path=None):
    assert type(action_space) == spaces.Discrete
    self._agent = DQNAgent(network, action_space)
    self._replay_memory = RemoteReplayMemory(
        is_server=True,
        memory_size=memory_size,
        memory_warmup_size=memory_warmup_size,
        ports=ports[:2])
    if init_model_path is not None:
      self._agent.load_params(
          torch.load(init_model_path,
                     map_location=lambda storage, loc: storage))
    self._model_params = self._agent.read_params()

    self._batch_size = batch_size
    self._mmc_beta = mmc_beta
    self._gradient_clipping = gradient_clipping
    self._adam_eps = adam_eps
    self._learning_rate = learning_rate
    self._target_update_interval = target_update_interval
    self._checkpoint_dir = checkpoint_dir
    self._checkpoint_interval = checkpoint_interval
    self._print_interval = print_interval
    self._discount = discount
    self._eps_start = eps_start
    self._eps_end = eps_end
    self._eps_decay_steps = eps_decay_steps
    self._eps_decay_steps2 = eps_decay_steps2
    self._epsilon = eps_start

    self._zmq_context = zmq.Context()
    self._reply_model_thread = Thread(
        target=self._reply_model, args=(self._zmq_context, ports[2]))
    self._reply_model_thread.start()

  def run(self):
    batch_queue = queue.Queue(8)
    batch_thread = Thread(target=self._prepare_batch,
                          args=(batch_queue, self._batch_size,))
    batch_thread.start()

    updates, loss, total_rollout_frames = 0, [], 0
    time_start = time.time()
    while True:
      updates += 1
      observation, next_observation, action, reward, done, mc_return = \
          batch_queue.get()
      self._epsilon = self._schedule_epsilon(updates)
      loss.append(self._agent.optimize_step(
          obs_batch=observation,
          next_obs_batch=next_observation,
          action_batch=action,
          reward_batch=reward,
          done_batch=done,
          mc_return_batch=mc_return,
          discount=self._discount,
          mmc_beta=self._mmc_beta,
          gradient_clipping=self._gradient_clipping,
          adam_eps=self._adam_eps,
          learning_rate=self._learning_rate,
          target_update_interval=self._target_update_interval))
      self._model_params = self._agent.read_params()
      if updates % self._checkpoint_interval == 0:
        ckpt_path = os.path.join(self._checkpoint_dir,
                                 'checkpoint-%d' % updates)
        self._save_checkpoint(ckpt_path)
      if updates % self._print_interval == 0:
        time_elapsed = time.time() - time_start
        train_fps = self._print_interval * self._batch_size / time_elapsed
        rollout_fps = (self._replay_memory.total - total_rollout_frames) \
            / time_elapsed
        loss_mean = np.mean(loss)
        tprint("Update: %d	Train-fps: %.1f	Rollout-fps: %.1f	"
               "Loss: %.5f	Epsilon: %.5f	Time: %.1f" % (updates, train_fps,
               rollout_fps, loss_mean, self._epsilon, time_elapsed))
        time_start, loss = time.time(), []
        total_rollout_frames = self._replay_memory.total

  def _prepare_batch(self, batch_queue, batch_size):
    while True:
      transitions = self._replay_memory.sample(batch_size)
      batch = self._transitions_to_batch(transitions)
      batch_queue.put(batch)

  def _transitions_to_batch(self, transitions):
    batch = Transition(*zip(*transitions))
    observation = torch.from_numpy(np.stack(batch.observation))
    next_observation = torch.from_numpy(np.stack(batch.next_observation))
    reward = torch.FloatTensor(batch.reward)
    action = torch.LongTensor(batch.action)
    done = torch.Tensor(batch.done)
    mc_return = torch.FloatTensor(batch.mc_return)

    if torch.cuda.is_available():
      observation = observation.pin_memory()
      next_observation = next_observation.pin_memory()
      action = action.pin_memory()
      reward = reward.pin_memory()
      mc_return = mc_return.pin_memory()
      done = done.pin_memory()

    return observation, next_observation, action, reward, done, mc_return

  def _save_checkpoint(self, checkpoint_path):
    torch.save(self._model_params, checkpoint_path)

  def _schedule_epsilon(self, steps):
    if steps < self._eps_decay_steps:
      return self._eps_start - (self._eps_start - self._eps_end) * \
          steps / self._eps_decay_steps
    elif steps < self._eps_decay_steps2:
      return self._eps_end - (self._eps_end - 0.01) * \
          (steps - self._eps_decay_steps) / self._eps_decay_steps2
    else:
      return 0.01

  def _reply_model(self, zmq_context, port):
    receiver = zmq_context.socket(zmq.REP)
    receiver.bind("tcp://*:%s" % port)
    while True:
      assert receiver.recv_string() == "request model"
      f = io.BytesIO()
      torch.save(self._model_params, f)
      receiver.send_pyobj(f.getvalue(), zmq.SNDMORE)
      receiver.send_pyobj(self._epsilon)
