from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from collections import deque
from queue import Queue
import queue
from threading import Thread
import time
import random
import joblib

import numpy as np
import tensorflow as tf
import zmq
from gym import spaces

from sc2learner.envs.spaces.mask_discrete import MaskDiscrete
from sc2learner.agents.utils_tf import explained_variance
from sc2learner.utils.utils import tprint


class Model(object):
  def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
               unroll_length, ent_coef, vf_coef, max_grad_norm, scope_name,
               value_clip=False):
    sess = tf.get_default_session()

    act_model = policy(sess, scope_name, ob_space, ac_space, nbatch_act, 1,
                       reuse=False)
    train_model = policy(sess, scope_name, ob_space, ac_space, nbatch_train,
                         unroll_length, reuse=True)

    A = tf.placeholder(shape=(nbatch_train,), dtype=tf.int32)
    ADV = tf.placeholder(tf.float32, [None])
    R = tf.placeholder(tf.float32, [None])
    OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
    OLDVPRED = tf.placeholder(tf.float32, [None])
    LR = tf.placeholder(tf.float32, [])
    CLIPRANGE = tf.placeholder(tf.float32, [])

    neglogpac = train_model.pd.neglogp(A)
    entropy = tf.reduce_mean(train_model.pd.entropy())

    vpred = train_model.vf
    vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED,
                                               -CLIPRANGE, CLIPRANGE)
    vf_losses1 = tf.square(vpred - R)
    if value_clip:
      vf_losses2 = tf.square(vpredclipped - R)
      vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
    else:
      vf_loss = .5 * tf.reduce_mean(vf_losses1)
    ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
    pg_losses = -ADV * ratio
    pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE,
                                         1.0 + CLIPRANGE)
    pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
    approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
    clipfrac = tf.reduce_mean(
        tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
    loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
    params = tf.trainable_variables(scope=scope_name)
    grads = tf.gradients(loss, params)
    if max_grad_norm is not None:
      grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
    grads = list(zip(grads, params))
    trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
    _train = trainer.apply_gradients(grads)
    new_params = [tf.placeholder(p.dtype, shape=p.get_shape()) for p in params]
    param_assign_ops = [p.assign(new_p) for p, new_p in zip(params, new_params)]

    def train(lr, cliprange, obs, returns, dones, actions, values, neglogpacs,
              states=None):
      advs = returns - values
      advs = (advs - advs.mean()) / (advs.std() + 1e-8)
      if isinstance(ac_space, MaskDiscrete):
        td_map = {train_model.X:obs[0], train_model.MASK:obs[-1], A:actions,
                  ADV:advs, R:returns, LR:lr, CLIPRANGE:cliprange,
                  OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}
      else:
        td_map = {train_model.X:obs, A:actions, ADV:advs, R:returns, LR:lr,
                  CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}
      if states is not None:
        td_map[train_model.STATE] = states
        td_map[train_model.DONE] = dones
      return sess.run(
        [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
        td_map
      )[:-1]
    self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy',
                       'approxkl', 'clipfrac']

    def save(save_path):
      joblib.dump(read_params(), save_path)

    def load(load_path):
      loaded_params = joblib.load(load_path)
      load_params(loaded_params)

    def read_params():
      return sess.run(params)

    def load_params(loaded_params):
      sess.run(param_assign_ops,
               feed_dict={p : v for p, v in zip(new_params, loaded_params)})

    self.train = train
    self.train_model = train_model
    self.act_model = act_model
    self.step = act_model.step
    self.value = act_model.value
    self.initial_state = act_model.initial_state
    self.save = save
    self.load = load
    self.read_params = read_params
    self.load_params = load_params

    tf.global_variables_initializer().run(session=sess)


class PPOActor(object):

  def __init__(self, env, policy, unroll_length, gamma, lam, queue_size=1,
               enable_push=True, learner_ip="localhost", port_A="5700",
               port_B="5701"):
    self._env = env
    self._unroll_length = unroll_length
    self._lam = lam
    self._gamma = gamma
    self._enable_push = enable_push

    self._model = Model(policy=policy,
                        scope_name="model",
                        ob_space=env.observation_space,
                        ac_space=env.action_space,
                        nbatch_act=1,
                        nbatch_train=unroll_length,
                        unroll_length=unroll_length,
                        ent_coef=0.01,
                        vf_coef=0.5,
                        max_grad_norm=0.5)
    self._obs = env.reset()
    self._state = self._model.initial_state
    self._done = False
    self._cum_reward = 0

    self._zmq_context = zmq.Context()
    self._model_requestor = self._zmq_context.socket(zmq.REQ)
    self._model_requestor.connect("tcp://%s:%s" % (learner_ip, port_A))
    if enable_push:
      self._data_queue = Queue(queue_size)
      self._push_thread = Thread(target=self._push_data, args=(
          self._zmq_context, learner_ip, port_B, self._data_queue))
      self._push_thread.start()

  def run(self):
    while True:
      # fetch model
      t = time.time()
      self._update_model()
      tprint("Update model time: %f" % (time.time() - t))
      t = time.time()
      # rollout
      unroll = self._nstep_rollout()
      if self._enable_push:
        if self._data_queue.full(): tprint("[WARN]: Actor's queue is full.")
        self._data_queue.put(unroll)
        tprint("Rollout time: %f" % (time.time() - t))

  def _nstep_rollout(self):
    mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = \
        [],[],[],[],[],[]
    mb_states, episode_infos = self._state, []
    for _ in range(self._unroll_length):
      action, value, self._state, neglogpac = self._model.step(
          transform_tuple(self._obs, lambda x: np.expand_dims(x, 0)),
          self._state,
          np.expand_dims(self._done, 0))
      mb_obs.append(transform_tuple(self._obs, lambda x: x.copy()))
      mb_actions.append(action[0])
      mb_values.append(value[0])
      mb_neglogpacs.append(neglogpac[0])
      mb_dones.append(self._done)
      self._obs, reward, self._done, info = self._env.step(action[0])
      self._cum_reward += reward
      if self._done:
        self._obs = self._env.reset()
        self._state = self._model.initial_state
        episode_infos.append({'r': self._cum_reward})
        self._cum_reward = 0
      mb_rewards.append(reward)
    if isinstance(self._obs, tuple):
      mb_obs = tuple(np.asarray(obs, dtype=self._obs[0].dtype)
                     for obs in zip(*mb_obs))
    else:
      mb_obs = np.asarray(mb_obs, dtype=self._obs.dtype)
    mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
    mb_actions = np.asarray(mb_actions)
    mb_values = np.asarray(mb_values, dtype=np.float32)
    mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
    mb_dones = np.asarray(mb_dones, dtype=np.bool)
    last_values = self._model.value(
        transform_tuple(self._obs, lambda x: np.expand_dims(x, 0)),
        self._state,
        np.expand_dims(self._done, 0))
    mb_returns = np.zeros_like(mb_rewards)
    mb_advs = np.zeros_like(mb_rewards)
    last_gae_lam = 0
    for t in reversed(range(self._unroll_length)):
      if t == self._unroll_length - 1:
        next_nonterminal = 1.0 - self._done
        next_values = last_values[0]
      else:
        next_nonterminal = 1.0 - mb_dones[t + 1]
        next_values = mb_values[t + 1]
      delta = mb_rewards[t] + self._gamma * next_values * next_nonterminal - \
          mb_values[t]
      mb_advs[t] = last_gae_lam = delta + self._gamma * self._lam * \
          next_nonterminal * last_gae_lam
    mb_returns = mb_advs + mb_values
    return (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs,
            mb_states, episode_infos)

  def _push_data(self, zmq_context, learner_ip, port_B, data_queue):
    sender = zmq_context.socket(zmq.PUSH)
    sender.setsockopt(zmq.SNDHWM, 1)
    sender.setsockopt(zmq.RCVHWM, 1)
    sender.connect("tcp://%s:%s" % (learner_ip, port_B))
    while True:
      data = data_queue.get()
      sender.send_pyobj(data)

  def _update_model(self):
      self._model_requestor.send_string("request model")
      self._model.load_params(self._model_requestor.recv_pyobj())


class PPOLearner(object):

  def __init__(self, env, policy, unroll_length, lr, clip_range, batch_size,
               ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, queue_size=8,
               print_interval=100, save_interval=10000, learn_act_speed_ratio=0,
               unroll_split=8, save_dir=None, init_model_path=None,
               port_A="5700", port_B="5701"):
    assert isinstance(env.action_space, spaces.Discrete)
    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(clip_range, float): clip_range = constfn(clip_range)
    else: assert callable(clip_range)
    self._lr = lr
    self._clip_range=clip_range
    self._batch_size = batch_size
    self._unroll_length = unroll_length
    self._print_interval = print_interval
    self._save_interval = save_interval
    self._learn_act_speed_ratio = learn_act_speed_ratio
    self._save_dir = save_dir

    self._model = Model(policy=policy,
                        scope_name="model",
                        ob_space=env.observation_space,
                        ac_space=env.action_space,
                        nbatch_act=1,
                        nbatch_train=batch_size * unroll_length,
                        unroll_length=unroll_length,
                        ent_coef=ent_coef,
                        vf_coef=vf_coef,
                        max_grad_norm=max_grad_norm)
    if init_model_path is not None: self._model.load(init_model_path)
    self._model_params = self._model.read_params()
    self._unroll_split = unroll_split if self._model.initial_state is None else 1
    assert self._unroll_length % self._unroll_split == 0
    self._data_queue = deque(maxlen=queue_size * self._unroll_split)
    self._data_timesteps = deque(maxlen=200)
    self._episode_infos = deque(maxlen=5000)
    self._num_unrolls = 0

    self._zmq_context = zmq.Context()
    self._pull_data_thread = Thread(
        target=self._pull_data,
        args=(self._zmq_context, self._data_queue, self._episode_infos,
              self._unroll_split, port_B)
    )
    self._pull_data_thread.start()
    self._reply_model_thread = Thread(
        target=self._reply_model, args=(self._zmq_context, port_A))
    self._reply_model_thread.start()

  def run(self):
    #while len(self._data_queue) < self._data_queue.maxlen: time.sleep(1)
    while len(self._episode_infos) < self._episode_infos.maxlen / 2:
      time.sleep(1)

    batch_queue = Queue(4)
    batch_threads = [
        Thread(target=self._prepare_batch,
               args=(self._data_queue, batch_queue,
                     self._batch_size * self._unroll_split))
        for _ in range(8)
    ]
    for thread in batch_threads:
      thread.start()

    updates, loss = 0, []
    time_start = time.time()
    while True:
      while (self._learn_act_speed_ratio > 0 and
          updates * self._batch_size >= \
          self._num_unrolls * self._learn_act_speed_ratio):
        time.sleep(0.001)
      updates += 1
      lr_now = self._lr(updates)
      clip_range_now = self._clip_range(updates)

      batch = batch_queue.get()
      obs, returns, dones, actions, values, neglogpacs, states = batch
      loss.append(self._model.train(lr_now, clip_range_now, obs, returns, dones,
                                    actions, values, neglogpacs, states))
      self._model_params = self._model.read_params()

      if updates % self._print_interval == 0:
        loss_mean = np.mean(loss, axis=0)
        batch_steps = self._batch_size * self._unroll_length
        time_elapsed = time.time() - time_start
        train_fps = self._print_interval * batch_steps / time_elapsed
        rollout_fps = len(self._data_timesteps) * self._unroll_length  / \
            (time.time() - self._data_timesteps[0])
        var = explained_variance(values, returns)
        avg_reward = safemean([info['r'] for info in self._episode_infos])
        tprint("Update: %d	Train-fps: %.1f	Rollout-fps: %.1f	"
               "Explained-var: %.5f	Avg-reward %.2f	Policy-loss: %.5f	"
               "Value-loss: %.5f	Policy-entropy: %.5f	Approx-KL: %.5f	"
               "Clip-frac: %.3f	Time: %.1f" % (updates, train_fps, rollout_fps,
               var, avg_reward, *loss_mean[:5], time_elapsed))
        time_start, loss = time.time(), []

      if self._save_dir is not None and updates % self._save_interval == 0:
        os.makedirs(self._save_dir, exist_ok=True)
        save_path = os.path.join(self._save_dir, 'checkpoint-%d' % updates)
        self._model.save(save_path)
        tprint('Saved to %s.' % save_path)

  def _prepare_batch(self, data_queue, batch_queue, batch_size):
    while True:
      batch = random.sample(data_queue, batch_size)
      obs, returns, dones, actions, values, neglogpacs, states = zip(*batch)
      if isinstance(obs[0], tuple):
        obs = tuple(np.concatenate(ob) for ob in zip(*obs))
      else:
        obs = np.concatenate(obs)
      returns = np.concatenate(returns)
      dones = np.concatenate(dones)
      actions = np.concatenate(actions)
      values = np.concatenate(values)
      neglogpacs = np.concatenate(neglogpacs)
      states = np.concatenate(states) if states[0] is not None else None
      batch_queue.put((obs, returns, dones, actions, values, neglogpacs, states))

  def _pull_data(self, zmq_context, data_queue, episode_infos, unroll_split,
                 port_B):
    receiver = zmq_context.socket(zmq.PULL)
    receiver.setsockopt(zmq.RCVHWM, 1)
    receiver.setsockopt(zmq.SNDHWM, 1)
    receiver.bind("tcp://*:%s" % port_B)
    while True:
      data = receiver.recv_pyobj()
      if unroll_split > 1:
        data_queue.extend(list(zip(*(
            [list(zip(*transform_tuple(
                data[0], lambda x: np.split(x, unroll_split))))] + \
                [np.split(arr, unroll_split) for arr in data[1:-2]] + \
                [[data[-2] for _ in range(unroll_split)]]
        ))))
      else:
        data_queue.append(data[:-1])
      episode_infos.extend(data[-1])
      self._data_timesteps.append(time.time())
      self._num_unrolls += 1

  def _reply_model(self, zmq_context, port_A):
    receiver = zmq_context.socket(zmq.REP)
    receiver.bind("tcp://*:%s" % port_A)
    while True:
      msg = receiver.recv_string()
      assert msg == "request model"
      receiver.send_pyobj(self._model_params)


class PPOAgent(object):

  def __init__(self, env, policy, model_path=None):
    assert isinstance(env.action_space, spaces.Discrete)
    self._model = Model(policy=policy,
                        scope_name="model",
                        ob_space=env.observation_space,
                        ac_space=env.action_space,
                        nbatch_act=1,
                        nbatch_train=1,
                        unroll_length=1,
                        ent_coef=0.01,
                        vf_coef=0.5,
                        max_grad_norm=0.5)
    if model_path is not None:
      self._model.load(model_path)
    self._state = self._model.initial_state
    self._done = False

  def act(self, observation):
      action, value, self._state, _ = self._model.step(
          transform_tuple(observation, lambda x: np.expand_dims(x, 0)),
          self._state,
          np.expand_dims(self._done, 0))
      return action[0]

  def reset(self):
    self._state = self._model.initial_state


class PPOSelfplayActor(object):

  def __init__(self, env, policy, unroll_length, gamma, lam, model_cache_size,
               model_cache_prob, queue_size=1, prob_latest_opponent=0.0,
               init_opponent_pool_filelist=None, freeze_opponent_pool=False,
               enable_push=True, learner_ip="localhost", port_A="5700",
               port_B="5701"):
    assert isinstance(env.action_space, spaces.Discrete)
    self._env = env
    self._unroll_length = unroll_length
    self._lam = lam
    self._gamma = gamma
    self._prob_latest_opponent = prob_latest_opponent
    self._freeze_opponent_pool = freeze_opponent_pool
    self._enable_push = enable_push
    self._model_cache_prob = model_cache_prob

    self._model = Model(policy=policy,
                        scope_name="model",
                        ob_space=env.observation_space,
                        ac_space=env.action_space,
                        nbatch_act=1,
                        nbatch_train=unroll_length,
                        unroll_length=unroll_length,
                        ent_coef=0.01,
                        vf_coef=0.5,
                        max_grad_norm=0.5)
    self._oppo_model = Model(policy=policy,
                             scope_name="oppo_model",
                             ob_space=env.observation_space,
                             ac_space=env.action_space,
                             nbatch_act=1,
                             nbatch_train=unroll_length,
                             unroll_length=unroll_length,
                             ent_coef=0.01,
                             vf_coef=0.5,
                             max_grad_norm=0.5)
    self._obs, self._oppo_obs = env.reset()
    self._state = self._model.initial_state
    self._oppo_state = self._oppo_model.initial_state
    self._done = False
    self._cum_reward = 0

    self._model_cache = deque(maxlen=model_cache_size)
    if init_opponent_pool_filelist is not None:
      with open(init_opponent_pool_filelist, 'r') as f:
        for model_path in f.readlines():
          print(model_path)
          self._model_cache.append(joblib.load(model_path.strip()))
    self._latest_model = self._oppo_model.read_params()
    if len(self._model_cache) == 0:
      self._model_cache.append(self._latest_model)
    self._update_opponent()

    self._zmq_context = zmq.Context()
    self._model_requestor = self._zmq_context.socket(zmq.REQ)
    self._model_requestor.connect("tcp://%s:%s" % (learner_ip, port_A))
    if enable_push:
      self._data_queue = Queue(queue_size)
      self._push_thread = Thread(target=self._push_data, args=(
          self._zmq_context, learner_ip, port_B, self._data_queue))
      self._push_thread.start()

  def run(self):
    while True:
      t = time.time()
      self._update_model()
      tprint("Time update model: %f" % (time.time() - t))
      t = time.time()
      unroll = self._nstep_rollout()
      if self._enable_push:
        if self._data_queue.full(): tprint("[WARN]: Actor's queue is full.")
        self._data_queue.put(unroll)
        tprint("Time rollout: %f" % (time.time() - t))

  def _nstep_rollout(self):
    mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = \
        [],[],[],[],[],[]
    mb_states, episode_infos = self._state, []
    for _ in range(self._unroll_length):
      action, value, self._state, neglogpac = self._model.step(
          transform_tuple(self._obs, lambda x: np.expand_dims(x, 0)),
          self._state,
          np.expand_dims(self._done, 0))
      oppo_action, _, self._oppo_state, _ = self._oppo_model.step(
          transform_tuple(self._oppo_obs, lambda x: np.expand_dims(x, 0)),
          self._oppo_state,
          np.expand_dims(self._done, 0))
      mb_obs.append(transform_tuple(self._obs, lambda x: x.copy()))
      mb_actions.append(action[0])
      mb_values.append(value[0])
      mb_neglogpacs.append(neglogpac[0])
      mb_dones.append(self._done)
      (self._obs, self._oppo_obs), reward, self._done, info = self._env.step(
        [action[0], oppo_action[0]])
      self._cum_reward += reward
      if self._done:
        self._obs, self._oppo_obs = self._env.reset()
        self._state = self._model.initial_state
        self._oppo_state = self._oppo_model.initial_state
        self._update_opponent()
        episode_infos.append({'r': self._cum_reward})
        self._cum_reward = 0
      mb_rewards.append(reward)
    if isinstance(self._obs, tuple):
      mb_obs = tuple(np.asarray(obs, dtype=self._obs[0].dtype)
                     for obs in zip(*mb_obs))
    else:
      mb_obs = np.asarray(mb_obs, dtype=self._obs.dtype)
    mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
    mb_actions = np.asarray(mb_actions)
    mb_values = np.asarray(mb_values, dtype=np.float32)
    mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
    mb_dones = np.asarray(mb_dones, dtype=np.bool)
    last_values = self._model.value(
        transform_tuple(self._obs, lambda x: np.expand_dims(x, 0)),
        self._state,
        np.expand_dims(self._done, 0))
    mb_returns = np.zeros_like(mb_rewards)
    mb_advs = np.zeros_like(mb_rewards)
    last_gae_lam = 0
    for t in reversed(range(self._unroll_length)):
      if t == self._unroll_length - 1:
        next_nonterminal = 1.0 - self._done
        next_values = last_values[0]
      else:
        next_nonterminal = 1.0 - mb_dones[t + 1]
        next_values = mb_values[t + 1]
      delta = mb_rewards[t] + self._gamma * next_values * next_nonterminal - \
          mb_values[t]
      mb_advs[t] = last_gae_lam = delta + self._gamma * self._lam * \
          next_nonterminal * last_gae_lam
    mb_returns = mb_advs + mb_values
    return (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs,
            mb_states, episode_infos)

  def _push_data(self, zmq_context, learner_ip, port_B, data_queue):
    sender = zmq_context.socket(zmq.PUSH)
    sender.setsockopt(zmq.SNDHWM, 1)
    sender.setsockopt(zmq.RCVHWM, 1)
    sender.connect("tcp://%s:%s" % (learner_ip, port_B))
    while True:
      data = data_queue.get()
      sender.send_pyobj(data)

  def _update_model(self):
      self._model_requestor.send_string("request model")
      model_params = self._model_requestor.recv_pyobj()
      self._model.load_params(model_params)
      if (not self._freeze_opponent_pool and
          random.uniform(0, 1.0) < self._model_cache_prob):
        self._model_cache.append(model_params)
      self._latest_model = model_params

  def _update_opponent(self):
    if (random.uniform(0, 1.0) < self._prob_latest_opponent or
        len(self._model_cache) == 0):
      self._oppo_model.load_params(self._latest_model)
      tprint("Opponent updated with the current model.")
    else:
      model_params = random.choice(self._model_cache)
      self._oppo_model.load_params(model_params)
      tprint("Opponent updated with the previous model. %d models cached." %
          len(self._model_cache))


def constfn(val):
  def f(_):
    return val
  return f


def safemean(xs):
  return np.nan if len(xs) == 0 else np.mean(xs)


def transform_tuple(x, transformer):
  if isinstance(x, tuple):
    return tuple(transformer(a) for a in x)
  else:
    return transformer(x)
