# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# episode generation

import random
import bz2
import pickle
import json
import glob


class Replayer:
    def __init__(self, env, args):
        self.env = env
        self.args = args

        all_json_paths = glob.glob(self.args['record_dir'] + "/**/*.json.tar")
        info_json_paths = set(glob.glob(self.args['record_dir'] + "/**/*_info.json.tar"))
        self.record_paths = [path for path in all_json_paths if path not in info_json_paths]

    def _error_check(self, info):
        for player in self.env.players():
            if info[player]['status'] == 'ERROR':
                return True
        return False

    def replay(self, record, args):
        # episode replay
        moments = []
        steps = 0
        skip_base = random.randrange(self.env.frame_skip + 1)

        if self._error_check(record['steps'][0]):
            return None
        self.env.resets_info(record['steps'][0])

        for info in record['steps'][1:self.env.limit_steps]:
            if self._error_check(info):
                return None

            trained_step = (steps + skip_base) % (self.env.frame_skip + 1) == 0
            if trained_step:
                moment = {
                    'o': [self.env.observation(player) for player in self.env.players()],
                    'a': [info[player]['action'][0] for player in self.env.players()]
                }
                moments.append(bz2.compress(pickle.dumps([moment])))

            self.env.plays_info(info)
            steps += 1

        if len(moments) < 1:
            return None

        outcomes = self.env.outcome()

        replay = {
            'args': args, 'outcome': outcomes, 'steps': len(moments), 'moment': moments
        }

        return replay

    def _select_record(self):
        while True:
            path = random.choice(self.record_paths)
            with open(path) as f:
                record = json.load(f)
            if isinstance(record, str):
                record = json.loads(record)
            if record['statuses'][0] != "DONE" or record['statuses'][1] != "DONE":
                continue
            return record

    def execute(self, args):
        record = self._select_record()
        replay = self.replay(record, args)
        if replay is None:
            print('None episode in replay!')
        return replay
