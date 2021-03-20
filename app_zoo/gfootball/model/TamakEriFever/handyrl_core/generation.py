# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# episode generation

import random
import bz2
import pickle

import numpy as np

from .model import softmax


class Generator:
    def __init__(self, env, args):
        self.env = env
        self.args = args

    def generate(self, models, opening_models, args):
        # episode generation
        steps = 0
        skip_base = random.randrange(self.env.frame_skip + 1)
        moments = []
        prev_action_list = [0, 0]
        hidden = {}
        for player in self.env.players():
            if player in args['player']:
                hidden[player] = models[player].init_hidden()
            else:
                models[player].reset(self.env)

        err = self.env.reset(args)
        if err:
            return None

        opening_randomize = True if random.random() < self.args['randomized_start_rate'] else False
        opening_randomized_steps = random.randrange(self.args['randomized_start_max_steps']) if opening_randomize else 0

        while not self.env.terminal():
            err = self.env.chance()
            if err:
                return None
            if self.env.terminal():
                break

            action_list = []
            opening_phase = steps < opening_randomized_steps
            trained_step = (steps + skip_base) % (self.env.frame_skip + 1) == 0 and not opening_phase

            if trained_step:
                moment = {'o': [], 'p': [], 'pm': [], 'a': [], 'rt': [], 'v': [], 'vt': [], 'ert': [], 'rtt': []}
            else:
                moment = None

            for player in self.env.players():
                if opening_phase:
                    action = opening_models[player].action(self.env, player)
                else:
                    model = models[player]

                    if player in args['player']:
                        if moment is not None:
                            obs = self.env.observation(player)
                            p_, v, r, hidden[player] = model.inference(obs, hidden[player])
                            legal_actions = self.env.legal_actions(player)

                            pmask = np.ones_like(p_)
                            pmask[legal_actions] = 0
                            p = p_ - pmask * 1e32
                            action = random.choices(legal_actions, weights=softmax(p[legal_actions]))[0]

                            moment['o'].append(obs)
                            moment['p'].append(p)
                            moment['pm'].append(pmask)
                            moment['a'].append(action)
                            moment['v'].append(v[0])
                            moment['ert'].append(r[0])
                        else:
                            action = prev_action_list[player]
                    else:
                        action = model.action(self.env, player)

                action_list.append(action)

            err = self.env.plays(action_list)
            prev_action_list = action_list
            steps += 1
            if err:
                return None

            if moment is not None:
                rw = self.env.reward()
                moment['rw'] = [r for index, r in enumerate(rw) if index in args['player']]
                moment['rs'] = self.args['reward_reset'] and self.env.is_reset_state()

            if moment is not None:
                moments.append(moment)

        if len(moments) < 1:
            return None

        outcomes = [oc for index, oc in enumerate(self.env.outcome()) if index in args['player']]

        for index in range(len(args['player'])):
            ret = 0
            vt = outcomes[index]
            rtt = ret
            for i, m in reversed(list(enumerate(moments))):
                rew = m['rw'][index] or 0
                ret = rew if m['rs'] else (rew + self.args['gamma'] * ret)
                moments[i]['rt'].append(ret)
                moments[i]['vt'].append(vt)
                vt = (1 - self.args['lambda']) * m['v'][index] + self.args['lambda'] * vt
                rtt_rw = rew if m['rs'] else (rew + self.args['gamma'] * rtt)
                moments[i]['rtt'].append(rtt_rw)
                rtt = (1 - self.args['lambda']) * (m['ert'][index] or 0) + self.args['lambda'] * rtt_rw

        csteps = self.args['compress_steps']
        episode = {
            'args': args, 'outcome': outcomes, 'steps': len(moments),
            'moment': [bz2.compress(pickle.dumps(moments[i:i+csteps])) for i in range(0, len(moments), csteps)],
        }

        return episode

    def execute(self, models, opening_models, args):
        episode = self.generate(models, opening_models, args)
        if episode is None:
            print('None episode in generation!')
        return episode
