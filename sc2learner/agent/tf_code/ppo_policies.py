from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from sc2learner.envs.spaces.mask_discrete import MaskDiscrete
from sc2learner.agents.utils_tf import CategoricalPd
from sc2learner.agents.utils_tf import fc, lstm, batch_to_seq, seq_to_batch


class MlpPolicy(object):
    def __init__(self, sess, scope_name, ob_space, ac_space, nbatch, nsteps,
                 reuse=False):
        if isinstance(ac_space, MaskDiscrete):
            ob_space, mask_space = ob_space.spaces

        X = tf.placeholder(
            shape=(nbatch,) + ob_space.shape, dtype=tf.float32, name="x_screen")
        if isinstance(ac_space, MaskDiscrete):
            MASK = tf.placeholder(
                shape=(nbatch,) + mask_space.shape, dtype=tf.float32, name="mask")

        with tf.variable_scope(scope_name, reuse=reuse):
            x = tf.layers.flatten(X)
            pi_h1 = tf.tanh(fc(x, 'pi_fc1', nh=512, init_scale=np.sqrt(2)))
            pi_h2 = tf.tanh(fc(pi_h1, 'pi_fc2', nh=512, init_scale=np.sqrt(2)))
            pi_h3 = tf.tanh(fc(pi_h2, 'pi_fc3', nh=512, init_scale=np.sqrt(2)))
            vf_h1 = tf.tanh(fc(x, 'vf_fc1', nh=512, init_scale=np.sqrt(2)))
            vf_h2 = tf.tanh(fc(vf_h1, 'vf_fc2', nh=512, init_scale=np.sqrt(2)))
            vf_h3 = tf.tanh(fc(vf_h2, 'vf_fc3', nh=512, init_scale=np.sqrt(2)))
            vf = fc(vf_h3, 'vf', 1)[:, 0]
            pi_logit = fc(pi_h3, 'pi', ac_space.n, init_scale=0.01, init_bias=0.0)
            if isinstance(ac_space, MaskDiscrete):
                pi_logit -= (1 - MASK) * 1e30
            self.pd = CategoricalPd(pi_logit)

        action = self.pd.sample()
        neglogp = self.pd.neglogp(action)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            if isinstance(ac_space, MaskDiscrete):
                a, v, nl = sess.run([action, vf, neglogp], {X: ob[0], MASK: ob[-1]})
            else:
                a, v, nl = sess.run([action, vf, neglogp], {X: ob})
            return a, v, self.initial_state, nl

        def value(ob, *_args, **_kwargs):
            if isinstance(ac_space, MaskDiscrete):
                return sess.run(vf, {X: ob[0], MASK: ob[-1]})
            else:
                return sess.run(vf, {X: ob})

        self.X = X
        if isinstance(ac_space, MaskDiscrete):
            self.MASK = MASK
        self.vf = vf
        self.step = step
        self.value = value


class LstmPolicy(object):

    def __init__(self, sess, scope_name, ob_space, ac_space, nbatch,
                 unroll_length, nlstm=512, reuse=False):
        nenv = nbatch // unroll_length
        if isinstance(ac_space, MaskDiscrete):
            ob_space, mask_space = ob_space.spaces

        DONE = tf.placeholder(tf.float32, [nbatch])
        STATE = tf.placeholder(tf.float32, [nenv, nlstm * 2])
        X = tf.placeholder(
            shape=(nbatch,) + ob_space.shape, dtype=tf.float32, name="x_screen")
        if isinstance(ac_space, MaskDiscrete):
            MASK = tf.placeholder(
                shape=(nbatch,) + mask_space.shape, dtype=tf.float32, name="mask")

        with tf.variable_scope(scope_name, reuse=reuse):
            x = tf.layers.flatten(X)
            fc1 = tf.nn.relu(fc(x, 'fc1', 512))
            h = tf.nn.relu(fc(fc1, 'fc2', 512))
            xs = batch_to_seq(h, nenv, unroll_length)
            ms = batch_to_seq(DONE, nenv, unroll_length)
            h5, snew = lstm(xs, ms, STATE, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            vf = fc(h5, 'v', 1)
            pi_logit = fc(h5, 'pi', ac_space.n, init_scale=1.0, init_bias=0.0)
            if isinstance(ac_space, MaskDiscrete):
                pi_logit -= (1 - MASK) * 1e30
            self.pd = CategoricalPd(pi_logit)

        action = self.pd.sample()
        neglogp = self.pd.neglogp(action)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, done):
            if isinstance(ac_space, MaskDiscrete):
                return sess.run([action, vf, snew, neglogp],
                                {X: ob[0], MASK: ob[-1], STATE: state, DONE: done})
            else:
                return sess.run([action, vf, snew, neglogp],
                                {X: ob, STATE: state, DONE: done})

        def value(ob, state, done):
            if isinstance(ac_space, MaskDiscrete):
                return sess.run(vf, {X: ob[0], MASK: ob[-1], STATE: state, DONE: done})
            else:
                return sess.run(vf, {X: ob, STATE: state, DONE: done})

        self.X = X
        if isinstance(ac_space, MaskDiscrete):
            self.MASK = MASK
        self.DONE = DONE
        self.STATE = STATE
        self.vf = vf
        self.step = step
        self.value = value
