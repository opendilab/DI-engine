from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
from collections import deque
from threading import Thread
import random
import time

import zmq


Transition = namedtuple('Transition',
                        ('observation', 'action', 'reward', 'next_observation',
                         'done', 'mc_return'))


class LocalReplayMemory(object):
  def __init__(self, capacity):
    self._memory = deque(maxlen=capacity)
    self._total = 0

  def push(self, *args):
    self._memory.append(Transition(*args))
    self._total += 1

  def sample(self, batch_size):
    return random.sample(self._memory, batch_size)

  @property
  def total(self):
    return self._total


class RemoteReplayMemory(object):
  def __init__(self,
               is_server,
               memory_size,
               memory_warmup_size,
               block_size=128,
               send_freq=1.0,
               num_pull_threads=4,
               ports=("5700", "5701"),
               server_ip="localhost"):
    assert len(ports) == 2
    assert memory_warmup_size <= memory_size
    self._is_server = is_server
    self._memory_warmup_size = memory_warmup_size
    self._block_size = block_size

    if is_server:
      self._num_received, self._num_used, self._total = 0, 0, 0
      self._cache_blocks = deque(maxlen=memory_size // block_size)
      self._zmq_context = zmq.Context()

      self._receiver_threads = [Thread(target=self._server_proxy_worker,
                                       args=(self._zmq_context, ports,))]
      self._receiver_threads += [Thread(target=self._server_receiver_worker,
                                        args=(self._zmq_context, ports[1],))
                                 for _ in range(num_pull_threads)]
      for thread in self._receiver_threads: thread.start()
    else:
      self._memory = LocalReplayMemory(memory_size)
      self._memory_total_last = 0
      self._send_interval = int(block_size / send_freq)

      self._zmq_context = zmq.Context()
      self._sender = self._zmq_context.socket(zmq.PUSH)
      self._sender.connect("tcp://%s:%s" % (server_ip, ports[0]))

  def push(self, *args):
    assert not self._is_server, "push() cannot be called when is_server=True."
    self._memory.push(*args)
    if (self._memory.total >= self._memory_warmup_size and
        self._memory.total >= self._block_size and
        self._memory.total % self._send_interval == 0):
      block = self._memory.sample(self._block_size)
      memory_total = self._memory.total
      memory_delta = memory_total - self._memory_total_last
      self._memory_total_last = memory_total
      self._sender.send_pyobj((block, memory_delta))

  def sample(self, batch_size, reuse_ratio=1.0):
    assert self._is_server, "sample() cannot be called when is_server=False."
    while (self._num_used / reuse_ratio >= self._num_received or
        self._memory_warmup_size > len(self._cache_blocks) * self._block_size):
      time.sleep(0.001)
    batch = [random.choice(random.choice(self._cache_blocks))
             for _ in range(batch_size)]
    self._num_used += batch_size
    return batch

  @property
  def total(self):
    if self._is_server:
      return self._total
    else:
      return self._memory.total

  def _server_receiver_worker(self, zmq_context, port):
    receiver = zmq_context.socket(zmq.PULL)
    receiver.connect("tcp://localhost:%s" % port)
    while True:
      block, delta = receiver.recv_pyobj()
      self._cache_blocks.append(block)
      self._total += delta
      self._num_received += len(block)

  def _server_proxy_worker(self, zmq_context, ports):
    assert len(ports) == 2
    frontend = zmq_context.socket(zmq.PULL)
    frontend.bind("tcp://*:%s" % ports[0])
    backend = self._zmq_context.socket(zmq.PUSH)
    backend.bind("tcp://*:%s" % ports[1])
    zmq.proxy(frontend, backend)


if __name__ == '__main__':
  import sys
  import numpy as np

  job_name = sys.argv[1]
  if job_name == 'client':
    replay_memory = RemoteReplayMemory(
        is_server=False,
        memory_size=10000,
        memory_warmup_size=16)
    while True:
      obs, next_obs = np.array([1,2,3]), np.array([3,4,5])
      action, reward, done, mc_return = 1, 0.5, False, 0.01
      replay_memory.push(obs, action, reward, next_obs, done, mc_return)
  else:
    replay_memory = RemoteReplayMemory(
        is_server=True,
        memory_size=10000,
        memory_warmup_size=10000)
    while True:
      print(replay_memory.sample(8))
