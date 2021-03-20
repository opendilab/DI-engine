# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]
#
# Paper that proposed VTrace algorithm
# https://arxiv.org/abs/1802.01561
# Official code
# https://github.com/deepmind/scalable_agent/blob/6c0c8a701990fab9053fb338ede9c915c18fa2b1/vtrace.py

# training

import os
import time
import copy
import threading
import random
import signal
import bz2
import pickle
import yaml
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optim

from .environment import prepare_env, make_env
from .util import map_r, bimap_r, trimap_r, rotate, type_r
from .model import load_model, to_torch, to_gpu_or_not, RandomModel
from .model import DuelingNet as Model
from .connection import MultiProcessWorkers
from .connection import accept_socket_connections
from .worker import Workers


def make_batch(episodes, args):
    """Making training batch

    Args:
        episodes (Iterable): list of episodes
        args (dict): training configuration

    Returns:
        dict: PyTorch input and target tensors

    Note:
        Basic data shape is (T, B, P, ...) .
        (T is time length, B is batch size, P is player count)
    """

    obss, datum = [], []

    for ep in episodes:
        # target player and turn index
        moments = sum([pickle.loads(bz2.decompress(ms)) for ms in ep['moment']], [])
        st = random.randrange(1 + max(0, len(moments) - args['forward_steps']))  # change start turn by sequence length
        moments = moments[st:st+args['forward_steps']]

        players = [ep['target']]

        obs_not_none = next(filter(lambda x: x is not None, moments[0]['o']), None)
        obs_zeros = map_r(obs_not_none, lambda o: np.zeros_like(o))  # template for padding
        obs = [m['o'] for m in moments]
        obs = rotate(obs)  # (T, P, ..., ...) -> (P, ..., T, ...)
        obs = rotate(obs)  # (P, ..., T, ...) -> (..., T, P, ...)
        obs = bimap_r(obs_zeros, obs, lambda _, o: np.array(o)[:, players])

        oc = np.array(ep['outcome'], dtype=np.float32)[..., np.newaxis][:, players]

        scalar_zeros = [0] * len(moments[0]['a'])
        rew = np.array([m.get('rw', scalar_zeros) for m in moments], dtype=np.float32)[..., np.newaxis][:, players]
        ret = np.array([m.get('rt', scalar_zeros) for m in moments], dtype=np.float32)[..., np.newaxis][:, players]
        res = np.array([m.get('rs', 0) for m in moments], dtype=np.float32)[..., np.newaxis, np.newaxis]
        vt = np.array([moments[-1].get('vt', scalar_zeros)], dtype=np.float32)[..., np.newaxis][:, players]
        rtt = np.array([moments[-1].get('rtt', scalar_zeros)], dtype=np.float32)[..., np.newaxis][:, players]

        p_zeros = [np.zeros(args['action_length'])] * len(moments[0]['a'])
        p = np.array([m.get('p', p_zeros) for m in moments], dtype=np.float32)[:, players]
        pmsk = np.array([m.get('pm', p_zeros) for m in moments], dtype=np.float32)[:, players] * 1e32

        emsk = np.ones((len(moments), 1, 1), dtype=np.float32)

        act = np.array([m['a'] for m in moments], dtype=np.int32)[..., np.newaxis][:, players]

        sv = np.array([[[ep.get('replay', False)]]], dtype=np.float32)  # supervised flag

        # pad each array if step length is short
        if len(moments) < args['forward_steps']:
            pad_len = args['forward_steps'] - len(moments)
            obs = map_r(obs, lambda o: np.pad(o, [(0, pad_len)] + [(0, 0)] * (len(o.shape) - 1), 'constant', constant_values=0))
            rew = np.pad(rew, [(0, pad_len), (0, 0), (0, 0)], 'constant', constant_values=0)
            ret = np.pad(ret, [(0, pad_len), (0, 0), (0, 0)], 'constant', constant_values=0)
            res = np.pad(res, [(0, pad_len), (0, 0), (0, 0)], 'constant', constant_values=1)
            pmsk = np.pad(pmsk, [(0, pad_len), (0, 0), (0, 0)], 'constant', constant_values=0)
            emsk = np.pad(emsk, [(0, pad_len), (0, 0), (0, 0)], 'constant', constant_values=0)
            act = np.pad(act, [(0, pad_len), (0, 0), (0, 0)], 'constant', constant_values=0)
            p = np.pad(p, [(0, pad_len), (0, 0), (0, 0)], 'constant', constant_values=0)

        if len(oc) < args['forward_steps']:
            oc = np.pad(oc, [(0, args['forward_steps'] - len(oc)), (0, 0), (0, 0)], 'constant', constant_values=0)

        obss.append(obs)
        datum.append((pmsk, emsk, act, p, rew, ret, res, oc, sv, vt, rtt))

    pmsk, emsk, act, p, rew, ret, res, oc, sv, vt, rtt = zip(*datum)

    obs = to_torch(bimap_r(obs_zeros, rotate(obss), lambda _, o: np.array(o)))
    pmsk = to_torch(np.array(pmsk))
    emsk = to_torch(np.array(emsk))
    act = to_torch(np.array(act))
    p = to_torch(np.array(p))
    rew = to_torch(np.array(rew))
    ret = to_torch(np.array(ret))
    res = to_torch(np.array(res))
    oc = to_torch(np.array(oc))
    sv = to_torch(np.array(sv))
    vt = to_torch(np.array(vt))
    rtt = to_torch(np.array(rtt))

    return {
        'observation': obs,
        'pmask': pmsk, 'emask': emsk,
        'action': act, 'policy': p,
        'reward': rew, 'return': ret, 'reset': res, 'outcome': oc,
        'supervised': sv,
        'last_value': vt, 'last_return': rtt
    }


def forward_prediction(model, hidden, batch):
    """Forward calculation via neural network

    Args:
        model (torch.nn.Module): neural network
        hidden: initial hidden state (..., L, B, P, ...)
        batch (dict): training batch (output of make_batch() function)

    Returns:
        tuple: calculated policy and value
    """

    observations = batch['observation']  # (B, T, P, ...)

    if hidden is None:
        # feed-forward neural network
        obs = map_r(observations, lambda o: o.view(-1, *o.size()[3:]))
        t_policies, t_values, t_returns, _ = model(obs, None)
    else:
        # sequential computation with RNN
        t_policies, t_values, t_returns = [], [], []
        for t in range(batch['emask'].size(1)):
            obs = map_r(observations, lambda o: o[:, t].view(-1, *o.size()[3:]))  # (..., B * P, ...)
            hidden = map_r(hidden, lambda h: h[:, t].view(h.size(0), -1, *h.size()[3:]))  # (..., B * P, ...)
            t_policy, t_value, t_return, hidden = model(obs, hidden)
            t_policies.append(t_policy)
            t_values.append(t_value)
            t_returns.append(t_return)
        t_policies = torch.stack(t_policies, dim=1)
        t_values = torch.stack(t_values, dim=1)
        t_returns = torch.stack(t_returns, dim=1) if t_returns[0] is not None else None

    # mask valid target values
    t_policies = t_policies.view(*batch['action'].size()[:-1], t_policies.size(-1))
    t_policies = t_policies.mul(batch['emask']) - batch['pmask']

    # mask valid target values
    if t_values is not None:
        t_values = t_values.view(*batch['action'].size()[:-1], t_values.size(-1))
        t_values = t_values.mul(batch['emask'])

    # mask valid cumulative rewards
    if t_returns is not None:
        t_returns = t_returns.view(*batch['action'].size()[:-1], t_returns.size(-1))
        t_returns = t_returns.mul(batch['emask'])

    return t_policies, t_values, t_returns


def compose_losses(policies, values, returns, \
                   log_selected_policies, advantages, value_targets, return_targets, \
                   batch, args):
    """Caluculate loss value

    Returns:
        tuple: losses and statistic values and the number of training data
    """

    outcomes = batch['outcome']
    actions = batch['action']
    emasks = batch['emask']
    supervised = batch['supervised']

    turn_advantages = advantages.sum(-1, keepdim=True)
    rp_loss = -log_selected_policies * turn_advantages
    rv_loss = ((values - value_targets) ** 2) / 2 if values is not None else None
    rr_loss = F.smooth_l1_loss(returns, return_targets, reduction='none') if returns is not None else None

    sp_loss = -log_selected_policies
    sp_accuracy = torch.argmax(policies, dim=-1, keepdim=True) == actions
    sv_loss = ((values - outcomes) ** 2) / 2 if values is not None else None

    entropy = dist.Categorical(logits=policies).entropy()

    return rp_loss, rv_loss, rr_loss, sp_loss, sp_accuracy, sv_loss, entropy, emasks, supervised


def gather_losses(losses, args):
    rp_loss, rv_loss, rr_loss, sp_loss, sp_accuracy, sv_loss, entropy, emasks, supervised = losses

    losses = {'r': {}, 's': {}}
    counts = {}

    # reinforcement loss
    reinforce = 1 - supervised
    counts['r'] = emasks.mul(reinforce).sum()
    losses['r']['p'] = rp_loss.mul(emasks).mul(reinforce).sum()
    if rv_loss is not None:
        losses['r']['v'] = rv_loss.mul(emasks).mul(reinforce).sum()
    if rr_loss is not None:
        losses['r']['r'] = rr_loss.mul(emasks).mul(reinforce).sum()

    # supervided loss
    counts['s'] = emasks.mul(supervised).sum()
    losses['s']['sp'] = sp_loss.mul(emasks).mul(supervised).sum()
    losses['s']['spa'] = sp_accuracy.mul(emasks).mul(supervised).sum()
    if sv_loss is not None:
        losses['s']['sv'] = sv_loss.mul(emasks).mul(supervised).sum()

    # entropy regularization
    entropy = entropy.mul(emasks.squeeze(-1))
    losses['r']['ent'] = entropy.mul(reinforce.squeeze(-1)).sum()
    losses['s']['ent'] = entropy.mul(supervised.squeeze(-1)).sum()

    reinforcement_loss = losses['r']['p'] + losses['r'].get('v', 0) + losses['r'].get('r', 0)
    supervised_loss = (losses['s']['sp'] + losses['s'].get('sv', 0)) * args['supervised_weight']

    total_loss = reinforcement_loss + supervised_loss + entropy.sum() * -args['entropy_regularization']

    return losses, total_loss, counts


def vtrace_base(batch, model, hidden, args):
    t_policies, t_values, t_returns = forward_prediction(model, hidden, batch)
    actions = batch['action']
    clip_rho_threshold = 1.0
    clip_c_threshold = 1.0 / (args['lambda'] + 1e-6)

    log_selected_b_policies = F.log_softmax(batch['policy'], dim=-1).gather(-1, actions) * batch['emask']
    log_selected_t_policies = F.log_softmax(t_policies     , dim=-1).gather(-1, actions) * batch['emask']

    # thresholds of importance sampling
    log_rhos = (log_selected_t_policies.detach() - log_selected_b_policies).sum(dim=2, keepdim=True)

    rhos = torch.exp(log_rhos)
    clipped_rhos = torch.clamp(rhos, 0, clip_rho_threshold)
    cs = torch.clamp(rhos, 0, clip_c_threshold)
    values_nograd = t_values.detach() if t_values is not None else None
    returns_nograd = t_returns.detach() if t_returns is not None else None

    if values_nograd is not None:
        if values_nograd.size(2) == 2:  # two player zerosum game
            values_nograd_opponent = -torch.stack([values_nograd[:, :, 1], values_nograd[:, :, 0]], dim=2)
            values_nograd = (values_nograd + values_nograd_opponent) / 2

        values_nograd = values_nograd * batch['emask'] + batch['outcome'] * (1 - batch['emask'])

    return t_policies, t_values, t_returns, log_selected_t_policies, \
        values_nograd, returns_nograd, clipped_rhos, cs


def vtrace(batch, model, hidden, args):
    # IMPALA
    # https://github.com/deepmind/scalable_agent/blob/master/vtrace.py

    t_policies, t_values, t_returns, log_selected_t_policies, \
        values_nograd, returns_nograd, clipped_rhos, cs = \
        vtrace_base(batch, model, hidden, args)
    outcomes, returns, rewards, resets = batch['outcome'], batch['return'], batch['reward'], batch['reset']
    last_values, last_returns = batch['last_value'], batch['last_return']
    time_length = batch['emask'].size(1)

    # value: LAMBDA-TRACE
    if t_values is not None:
        lamb = args['lambda']
        mc_rate = args['monte_carlo_rate']
        final_values = outcomes[:, 0] * mc_rate + last_values[:, 0] * (1 - mc_rate)
        lambda_values = deque([final_values])
        upgo_values = deque([final_values])
        for i in range(time_length - 2, -1, -1):
            lambda_values.appendleft((1 - lamb) * values_nograd[:, i + 1] + lamb * lambda_values[0])
            upgo_values.appendleft(torch.max(values_nograd[:, i + 1], (1 - lamb) * values_nograd[:, i + 1] + lamb * upgo_values[0]))

        lambda_values = torch.stack(tuple(lambda_values), dim=1).contiguous()
        upgo_values = torch.stack(tuple(upgo_values), dim=1).contiguous()
        value_targets = lambda_values
        value_advantages = upgo_values - values_nograd
    else:
        value_targets = None
        value_advantages = 0

    # return: LAMBDA-TRACE
    if t_returns is not None:
        lamb = args['lambda']
        gamma = args['gamma']
        mc_rate = args['monte_carlo_rate']
        final_returns = returns[:, -1] * mc_rate + last_returns[:, 0] * (1 - mc_rate)
        lambda_returns = deque([final_returns])

        for i in range(time_length - 2, -1, -1):
            lambda_returns.appendleft(rewards[:, i] + gamma * (1 - resets[:, i])  * ((1 - lamb) * returns_nograd[:, i + 1] + lamb * lambda_returns[0]))

        lambda_returns = torch.stack(tuple(lambda_returns), dim=1).contiguous()
        return_targets = lambda_returns
        return_advantages = lambda_returns - returns_nograd
    else:
        return_targets = None
        return_advantages = 0

    # compute policy advantage
    advantages = clipped_rhos * (value_advantages + return_advantages) / 2

    return compose_losses(
        t_policies, t_values, t_returns,
        log_selected_t_policies, advantages, value_targets, return_targets,
        batch, args
    )


class TrainerModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def init_hidden(self, batch_shape):
        return self.model.init_hidden(batch_shape)

    def forward(self, batch, hidden, args):
        return vtrace(batch, self.model, hidden, args)


class Batcher:
    def __init__(self, args, episodes, replays, gpu):
        self.args = args
        self.episodes = episodes
        self.replays = replays
        self.gpu = gpu
        self.shutdown_flag = False

        self.workers = MultiProcessWorkers(
            self._worker, self._selector(), self.args['num_batchers'], postprocess=None,
            buffer_length=6, num_receivers=2
        )

    def _selector(self):
        replay_size = int(self.args['batch_size'] * self.args['replay_rate'])
        episode_size = self.args['batch_size'] - replay_size
        while True:
            yield [self.select_episode() for _ in range(episode_size)] + \
                [self.select_replay() for _ in range(replay_size)]

    def _worker(self, conn, bid):
        print('started batcher %d' % bid)
        while not self.shutdown_flag:
            episodes = conn.recv()
            batch = make_batch(episodes, self.args)
            conn.send((batch, 1))
        print('finished batcher %d' % bid)

    def run(self):
        self.workers.start()

    def select_episode(self):
        num_blocks_sent = (self.args['forward_steps'] // self.args['compress_steps']) + 1
        while True:
            ep_idx = random.randrange(min(len(self.episodes), self.args['maximum_episodes']))
            accept_rate = 1 - (len(self.episodes) - 1 - ep_idx) / self.args['maximum_episodes']
            if random.random() < accept_rate:
                ep = self.episodes[ep_idx]
                st = random.randrange(1 + max(0, len(ep['moment']) - num_blocks_sent))  # change start turn by sequence length
                ep_minimum = {
                    'target': random.randrange(len(ep['args']['player'])),
                    'outcome': [ep['outcome']],
                    'moment': ep['moment'][st:st+num_blocks_sent]
                }
                return ep_minimum

    def select_replay(self):
        moments, outcomes = [], []
        for _ in range(self.args['forward_steps']):
            rep_idx = random.randrange(min(len(self.replays), self.args['maximum_replays']))
            rep = self.replays[rep_idx]
            step = random.randrange(len(rep['moment']))
            moments.append(rep['moment'][step])
            outcomes.append(rep['outcome'])
        ep_like = {
            'replay': True,
            'target': random.randrange(self.args['num_players']),
            'outcome': outcomes,
            'moment': moments
        }
        return ep_like

    def batch(self):
        return self.workers.recv()

    def shutdown(self):
        self.shutdown_flag = True
        self.workers.shutdown()


class Trainer:
    def __init__(self, args, model):
        self.episodes = deque()
        self.replays = deque()
        self.args = args
        self.gpu = torch.cuda.device_count()
        self.model = model
        self.defalut_lr = 3e-8
        self.data_cnt_ema = self.args['batch_size'] * self.args['forward_steps']
        self.params = list(self.model.parameters())
        lr = self.defalut_lr * self.data_cnt_ema
        self.optimizer = optim.Adam(self.params, lr=lr, weight_decay=1e-5) if len(self.params) > 0 else None
        self.steps = 0
        self.started = False
        self.lock = threading.Lock()
        self.batcher = Batcher(self.args, self.episodes, self.replays, self.gpu)
        self.updated_model = None, 0
        self.update_flag = False
        self.shutdown_flag = False

    def update(self):
        if len(self.episodes) < self.args['minimum_episodes']:
            return None, 0  # return None before training
        self.update_flag = True
        while True:
            time.sleep(0.1)
            model, steps = self.recheck_update()
            if model is not None:
                break
        return model, steps

    def report_update(self, model, steps):
        self.lock.acquire()
        self.update_flag = False
        self.updated_model = model, steps
        self.lock.release()

    def recheck_update(self):
        self.lock.acquire()
        flag = self.update_flag
        self.lock.release()
        return (None, -1) if flag else self.updated_model

    def shutdown(self):
        self.shutdown_flag = True
        self.batcher.shutdown()

    def train(self):
        if self.optimizer is None:  # non-parametric model
            print()
            return

        batch_cnt, data_cnt, loss_sum, total_loss_sum = 0, {}, {}, 0
        train_model = self.model
        if self.gpu:
            if self.gpu > 1:
                train_model = nn.DataParallel(self.model)
            train_model.cuda()
        train_model.train()

        while data_cnt == 0 or not (self.update_flag or self.shutdown_flag):
            # episodes were only tuple of arrays
            batch = to_gpu_or_not(self.batcher.batch(), self.gpu)
            batch_size = batch['action'].size(1)
            player_count = batch['action'].size(2)
            hidden = to_gpu_or_not(self.model.init_hidden([batch_size, player_count]), self.gpu)

            losses = train_model(batch, hidden, self.args)
            losses, total_loss, counts = gather_losses(losses, self.args)

            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.params, 4.0)
            self.optimizer.step()

            batch_cnt += 1
            for key, cnt in counts.items():
                data_cnt[key] = data_cnt.get(key, 0) + cnt.item()

            total_loss_sum += total_loss.item()
            for key, loss in losses.items():
                loss_sum[key] = {k: loss_sum.get(key, {}).get(k, 0.0) + l.item() for k, l in loss.items()}

            self.steps += 1

        for key, loss in loss_sum.items():
            if data_cnt[key] > 0:
                print('loss-%s = %s' % (key, ' '.join([k + ':' + '%.3f' % (l / data_cnt[key]) for k, l in loss.items()])))

        data_cnt_sum = sum(list(data_cnt.values()))
        if data_cnt_sum > 0:
            print('loss-total = %s' % ('%.3f' % (total_loss_sum / data_cnt_sum)))

        self.data_cnt_ema = self.data_cnt_ema * 0.8 + data_cnt_sum / (1e-2 + batch_cnt) * 0.2
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.defalut_lr * self.data_cnt_ema / (1 + self.steps * 1e-5)
        model = self.model.model
        model.cpu()
        model.eval()
        return copy.deepcopy(model)

    def run(self):
        print('waiting training')
        while not self.shutdown_flag:
            if len(self.episodes) < self.args['minimum_episodes']:
                time.sleep(1)
                continue
            if not self.started:
                self.started = True
                self.batcher.run()
                print('started training')
            model = self.train()
            self.report_update(model, self.steps)
        print('finished training')


class Learner:
    def __init__(self, args):
        self.args = args
        random.seed(args['seed'])
        self.env = make_env(args['env'])
        eval_modify_rate = (args['update_episodes'] ** 0.85) / args['update_episodes']
        self.eval_rate = max(args['eval_rate'], eval_modify_rate)
        self.replay_rate = args['replay_rate']
        self.shutdown_flag = False

        frames_per_sec = args['env']['frames_per_sec'] / (1 + args['env']['frame_skip'])
        args['lambda'] = args['lambda_per_sec'] ** (1.0 / frames_per_sec)
        args['gamma'] = args['gamma_per_sec'] ** (1.0 / frames_per_sec)
        args['action_length'] = self.env.action_length()
        args['num_players'] = len(self.env.players())
        args['maximum_replays'] = args['maximum_episodes'] * args['replay_rate']

        # trained datum
        self.model_era = self.args.get('restart_epoch', 0)
        self.model_class = self.env.net() if hasattr(self.env, 'net') else Model
        train_model = TrainerModel(self.model_class(self.env, args))
        if self.model_era == 0:
            self.model = RandomModel(self.env)
        else:
            self.model = load_model(train_model.model, self.model_path(self.model_era))

        # generated datum
        self.generation_results = {}
        self.num_episodes = 0

        # replayed datum
        self.num_replays = 0

        # evaluated datum
        self.results = {}
        self.results_per_opponent = {}
        self.num_results = 0

        # multiprocess or remote connection
        self.workers = Workers(args)

        # thread connection
        self.trainer = Trainer(args, train_model)

    def shutdown(self):
        self.shutdown_flag = True
        self.trainer.shutdown()
        self.workers.shutdown()
        for thread in self.threads:
            thread.join()

    def model_path(self, model_id):
        return os.path.join('models', str(model_id) + '.pth')

    def latest_model_path(self):
        return os.path.join('models', 'latest.pth')

    def update_model(self, model, steps):
        # get latest model and save it
        print('updated model(%d)' % steps)
        self.model_era += 1
        self.model = model
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), self.model_path(self.model_era))
        torch.save(model.state_dict(), self.latest_model_path())

    def feed_episodes(self, episodes):
        # analyze generated episodes
        for episode in episodes:
            if episode is None:
                continue
            if self.model_era not in self.generation_results:
                self.generation_results[self.model_era] = 0, 0, 0
            for p in episode['args']['player']:
                result = episode['outcome'][p]
                n, r, r2 = self.generation_results[self.model_era]
                self.generation_results[self.model_era] = n + 1, r + result, r2 + result ** 2

        # store generated episodes
        self.trainer.episodes.extend([e for e in episodes if e is not None])
        while len(self.trainer.episodes) > self.args['maximum_episodes']:
            self.trainer.episodes.popleft()

    def feed_replays(self, replays):
        # store loaded replays
        self.trainer.replays.extend([r for r in replays if r is not None])
        while len(self.trainer.replays) > self.args['maximum_replays']:
            self.trainer.replays.popleft()

    def feed_results(self, results):
        # store evaluation results
        for model_id, result in results:
            if result is None:
                continue
            opp, reward = result

            if model_id not in self.results:
                self.results[model_id] = {}
            if reward not in self.results[model_id]:
                self.results[model_id][reward] = 0
            self.results[model_id][reward] += 1

            if model_id not in self.results_per_opponent:
                self.results_per_opponent[model_id] = {}
            if opp not in self.results_per_opponent[model_id]:
                self.results_per_opponent[model_id][opp] = {}
            if reward not in self.results_per_opponent[model_id][opp]:
                self.results_per_opponent[model_id][opp][reward] = 0
            self.results_per_opponent[model_id][opp][reward] += 1

    def update(self):
        # call update to every component
        print()
        print('epoch %d' % self.model_era)
        if self.model_era not in self.results:
            print('win rate = Nan (0)')
        else:
            def output_wp(name, distribution):
                results = {k: distribution[k] for k in sorted(distribution.keys(), reverse=True)}
                # output evaluation results
                n, win = 0, 0.0
                for r, cnt in results.items():
                    n += cnt
                    win += (r + 1) / 2 * cnt
                print('win rate (%s) = %.3f (%.1f / %d)' % (name, win / n, win, n))

            output_wp("total", self.results[self.model_era])
            for key in sorted(list(self.results_per_opponent[self.model_era])):
                output_wp(key, self.results_per_opponent[self.model_era][key])
        if self.model_era not in self.generation_results:
            print('generation stats = Nan (0)')
        else:
            n, r, r2 = self.generation_results[self.model_era]
            mean = r / (n + 1e-6)
            std = (r2 / (n + 1e-6) - mean ** 2) ** 0.5
            print('generation stats = %.3f +- %.3f' % (mean, std))
        model, steps = self.trainer.update()
        if model is None:
            model = self.model
        self.update_model(model, steps)

    def server(self):
        # central conductor server
        # returns as list if getting multiple requests as list
        print('started server')
        prev_update_episodes = self.args['minimum_episodes']
        while True:
            # no update call before storings minimum number of episodes + 1 age
            next_update_episodes = prev_update_episodes + self.args['update_episodes']
            while not self.shutdown_flag and self.num_episodes < next_update_episodes:
                conn, (req, data) = self.workers.recv()
                multi_req = isinstance(data, list)
                if not multi_req:
                    data = [data]
                send_data = []

                if req == 'args':
                    for _ in data:
                        args = {'model_id': {}}

                        # decide role
                        if self.num_results < self.eval_rate * self.num_episodes:
                            args['role'] = 'e'
                        elif self.num_replays < self.replay_rate * self.num_episodes:
                            args['role'] = 'r'
                        else:
                            args['role'] = 'g'

                        if args['role'] == 'g':
                            # genatation configuration
                            args['player'] = [0]
                            args['model_id'][0] = self.model_era
                            args['model_id'][1] = -1

                            args['limit_rate'] = random.random()  # shorter episode in generation

                            self.num_episodes += 1
                            if self.num_episodes % 100 == 0:
                                print(self.num_episodes, end=' ', flush=True)

                        elif args['role'] == 'r':
                            # replay configuration
                            args['player'] = [0, 1]

                            self.num_replays += 1

                        elif args['role'] == 'e':
                            # evaluation configuration
                            args['player'] = [0]
                            args['model_id'][0] = self.model_era
                            args['model_id'][1] = -1

                            self.num_results += 1

                        send_data.append(args)

                elif req == 'episode':
                    # report generated episodes
                    self.feed_episodes(data)
                    send_data = [None] * len(data)

                elif req == 'replay':
                    # report generated episodes
                    self.feed_replays(data)
                    send_data = [None] * len(data)

                elif req == 'result':
                    # report evaluation results
                    self.feed_results(data)
                    send_data = [None] * len(data)

                elif req == 'model':
                    for model_id in data:
                        if model_id == self.model_era:
                            model = self.model
                        else:
                            try:
                                model = self.model_class(self.env, self.args)
                                model = load_model(self.model_path(model_id))
                            except:
                                # return latest model if failed to load specified model
                                pass
                        send_data.append(model)

                if not multi_req and len(send_data) == 1:
                    send_data = send_data[0]
                self.workers.send(conn, send_data)
            prev_update_episodes = next_update_episodes
            self.update()
        print('finished server')

    def entry_server(self):
        port = 9999
        print('started entry server %d' % port)
        conn_acceptor = accept_socket_connections(port=port, timeout=0.3)
        while not self.shutdown_flag:
            conn = next(conn_acceptor)
            if conn is not None:
                entry_args = conn.recv()
                print('accepted entry from %s!' % entry_args['host'])
                args = copy.deepcopy(self.args)
                args['worker'] = entry_args
                conn.send(args)
                conn.close()
        print('finished entry server')

    def run(self):
        try:
            # open threads
            self.threads = [threading.Thread(target=self.trainer.run)]
            if self.args['remote']:
                self.threads.append(threading.Thread(target=self.entry_server))
            for thread in self.threads:
                thread.start()
            # open generator, evaluator
            self.workers.run()
            self.server()

        finally:
            self.shutdown()


def train_main(args):
    train_args = args['train_args']
    train_args['remote'] = False

    env_args = args['env_args']
    train_args['env'] = env_args

    prepare_env(env_args)  # preparing environment is needed in stand-alone mode
    learner = Learner(train_args)
    learner.run()

def train_server_main(args):
    train_args = args['train_args']
    train_args['remote'] = True

    env_args = args['env_args']
    train_args['env'] = env_args

    learner = Learner(train_args)
    learner.run()
