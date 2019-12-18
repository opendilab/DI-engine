from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


class Pd(object):

  def neglogp(self, x):
    raise NotImplementedError

  def entropy(self):
    raise NotImplementedError

  def sample(self):
    raise NotImplementedError

  @classmethod
  def fromlogits(cls, logits):
    return cls(logits)


class CategoricalPd(Pd):

  def __init__(self, logits):
    self.logits = logits

  def neglogp(self, x):
    one_hot_actions = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
    return tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                   labels=one_hot_actions)

  def entropy(self):
    a = self.logits - tf.reduce_max(self.logits, axis=-1, keep_dims=True)
    ea = tf.exp(a)
    z = tf.reduce_sum(ea, axis=-1, keep_dims=True)
    p = ea / z
    return tf.reduce_sum(p * (tf.log(z) - a), axis=-1)

  def sample(self):
    u = tf.random_uniform(tf.shape(self.logits))
    return tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)


def fc(x, scope, nh, init_scale=1.0, init_bias=0.0):
  with tf.variable_scope(scope):
    nin = x.get_shape()[1].value
    w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
    b = tf.get_variable("b", [nh],
                        initializer=tf.constant_initializer(init_bias))
    return tf.matmul(x, w) + b


def lstm(xs, ms, s, scope, nh, init_scale=1.0):
  nbatch, nin = [v.value for v in xs[0].get_shape()]
  nsteps = len(xs)
  with tf.variable_scope(scope):
    wx = tf.get_variable("wx", [nin, nh*4], initializer=ortho_init(init_scale))
    wh = tf.get_variable("wh", [nh, nh*4], initializer=ortho_init(init_scale))
    b = tf.get_variable("b", [nh*4], initializer=tf.constant_initializer(0.0))

  c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
  for idx, (x, m) in enumerate(zip(xs, ms)):
    c = c * (1 - m)
    h = h * (1 - m)
    z = tf.matmul(x, wx) + tf.matmul(h, wh) + b
    i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
    i = tf.nn.sigmoid(i)
    f = tf.nn.sigmoid(f)
    o = tf.nn.sigmoid(o)
    u = tf.tanh(u)
    c = f * c + i * u
    h = o * tf.tanh(c)
    xs[idx] = h
  s = tf.concat(axis=1, values=[c, h])
  return xs, s


def batch_to_seq(h, nbatch, nsteps, flat=False):
  if flat: h = tf.reshape(h, [nbatch, nsteps])
  else: h = tf.reshape(h, [nbatch, nsteps, -1])
  return [tf.squeeze(v, [1])
          for v in tf.split(axis=1, num_or_size_splits=nsteps, value=h)]


def seq_to_batch(h, flat = False):
  shape = h[0].get_shape().as_list()
  if not flat:
    assert len(shape) > 1
    nh = h[0].get_shape()[-1].value
    return tf.reshape(tf.concat(axis=1, values=h), [-1, nh])
  else:
    return tf.reshape(tf.stack(values=h, axis=1), [-1])


def ortho_init(scale=1.0):

  def _ortho_init(shape, dtype, partition_info=None):
    shape = tuple(shape)
    if len(shape) == 2: flat_shape = shape
    elif len(shape) == 4: flat_shape = (np.prod(shape[:-1]), shape[-1]) #NHWC
    else: raise NotImplementedError
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return (scale * q[:shape[0], :shape[1]]).astype(np.float32)

  return _ortho_init


def explained_variance(ypred,y):
  assert y.ndim == 1 and ypred.ndim == 1
  var_y = np.var(y)
  return np.nan if var_y == 0 else 1 - np.var(y - ypred) / var_y
