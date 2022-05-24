import os
import shutil
import argparse

from distar.ctools.utils import read_config, deep_merge_dicts
from distar.actor import Actor
import torch
import random
import time

class TestDIstarEnv:
    def __init__(self):

        cfg = read_config('C:/Users/hjs/DI-engine/dizoo/distar/envs/test_distar_config.yaml')
        self._whole_cfg = cfg
        self._whole_cfg.env.map_name = 'KingsCove'

    def _inference_loop(self, job={}):
        from dizoo.distar.envs import DIStarEnv
        import traceback

        torch.set_num_threads(1)

        self._env = DIStarEnv(self._whole_cfg)

        with torch.no_grad():
            for _ in range(5):
                try:
                    observations, game_info, map_name = self._env.reset()

                    for iter in range(1000):  # one episode loop
                        # agent step
                        actions = self._env.random_action(observations)
                        # env step
                        next_observations, reward, done = self._env.step(actions)
                        if not done:
                            observations = next_observations
                        else:
                            break
                        
                except Exception as e:
                    print('[EPISODE LOOP ERROR]', e, flush=True)
                    print(''.join(traceback.format_tb(e.__traceback__)), flush=True)
                    self._env.close()
            self._env.close()

if __name__ == '__main__':

    ## main
    if os.path.exists(r'C:\Program Files (x86)\StarCraft II'):
        sc2path = r'C:\Program Files (x86)\StarCraft II'
    elif os.path.exists('/Applications/StarCraft II'):
        sc2path = '/Applications/StarCraft II'
    else:
        assert 'SC2PATH' in os.environ.keys(), 'please add StarCraft2 installation path to your environment variables!'
        sc2path = os.environ['SC2PATH']
        assert os.path.exists(sc2path), 'SC2PATH: {} does not exist!'.format(sc2path)
    if not os.path.exists(os.path.join(sc2path, 'Maps/Ladder2019Season2')):
        shutil.copytree(os.path.join(os.path.dirname(__file__), '../envs/maps/Ladder2019Season2'), os.path.join(sc2path, 'Maps/Ladder2019Season2'))

    ## actor_run
    actor = TestDIstarEnv()
    actor._inference_loop()