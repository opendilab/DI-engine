import os
from collections import namedtuple
from ding.envs import BaseEnv
from typing import Optional, Any, List
from easydict import EasyDict
import numpy as np
from ding.utils.data import default_decollate
from ding.utils import build_logger, EasyTimer, SERIAL_COLLECTOR_REGISTRY


@SERIAL_COLLECTOR_REGISTRY.register('alphazero')
class AlphazeroCollector:

    def __init__(
        self,
        cfg: EasyDict,
        env: BaseEnv = None,
        policy: namedtuple = None,
        tb_logger: 'SummaryWriter' = None,  # noqa
        exp_name: Optional[str] = 'default_experiment',
        instance_name: Optional[str] = 'collector'
    ):
        self._cfg = cfg
        self._env = env
        self._policy = policy
        self._collect_n_episode = self._cfg.collect_n_episode
        self._max_moves = self._cfg.max_moves
        self._exp_name = exp_name
        self._instance_name = instance_name
        self._collect_print_freq = self._cfg.print_freq
        # self._deepcopy_obs = self._cfg.deepcopy_obs
        # self._transform_obs = self._cfg.transform_obs
        self._use_augmentation = self._cfg.augmentation
        self._timer = EasyTimer()
        self._end_flag = False
        if tb_logger is not None:
            self._logger, _ = build_logger(
                path='./{}/log/{}'.format(self._exp_name, self._instance_name), name=self._instance_name, need_tb=False
            )
            self._tb_logger = tb_logger
        else:
            self._logger, self._tb_logger = build_logger(
                path='./{}/log/{}'.format(self._exp_name, self._instance_name), name=self._instance_name
            )
        self._iter_count = 0
        self.envstep = 0
        self.winner_list = []

    def collect(self,
                n_episode: Optional[int] = None,
                train_iter: int = 0,
                policy_kwargs: Optional[dict] = None) -> List[Any]:
        if n_episode is None:
            n_episode = self._collect_n_episode
        data_all = []
        for i in range(n_episode):
            winner, data = self._self_play()
            self.winner_list.append(winner)
            data_all.extend(data)
        self._iter_count += 1
        if self._iter_count % self._collect_print_freq == 0:
            winrate = ((np.array(self.winner_list) == 1) + 0.5 * (np.array(self.winner_list) == -1)).mean()
            self._logger.info(f'winrate_list:{self.winner_list}')
            self._logger.info(f'player1 winrate:{winrate}')
            self._logger.info(f'player2 winrate:{1-winrate}')
            self.winner_list.clear()

        return data_all

    def _self_play(self, ):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.env.reset()
        states, mcts_probs, current_players = [], [], []
        done = False
        while not done and len(states) < self._max_moves:
            action, move_probs = self._policy.forward(self.env)
            # store the data
            states.append(self.env.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.env.current_player)
            # perform an action
            self.env.do_action(action)
            done, winner = self.env.game_end()
            if done:
                # winner from the perspective of the current player of each state
                winners = np.zeros(len(current_players), dtype=np.float32)
                if winner != -1:
                    winners[np.array(current_players) == winner] = 1.0
                    winners[np.array(current_players) != winner] = -1.0
                if winner != -1:
                    print("Game end. Winner is player:", winner)
                else:
                    print("Game end. Tie")
                # mini_batch = {}
                # mini_batch['state'] = states
                # mini_batch['mcts_prob'] = mcts_probs
                # mini_batch['winner'] = winner
                # mini_batch = default_decollate(mini_batch)
                data = [
                    {
                        'state': state,
                        'mcts_prob': mcts_prob,
                        'winner': winner
                    } for state, mcts_prob, winner in zip(states, mcts_probs, winners)
                ]
                # return winner, zip(states,mcts_probs,winners)
                self.envstep += len(data)
                if self._use_augmentation:
                    data = self.get_equi_data(data)
                return winner, data

    def get_equi_data(self, data):
        if hasattr(self._env, 'get_equi_data'):
            return self.env.get_equi_data(data)
        else:
            return data

    @property
    def env(self):
        return self._env

    @env.setter
    def env(self, _env):
        self._env = _env

    def close(self) -> None:
        """
        Overview:
            Close the collector. If end_flag is False, close the environment, flush the tb_logger\
                and close the tb_logger.
        """
        if self._end_flag:
            return
        self._end_flag = True
        self._env.close()
        self._tb_logger.flush()
        self._tb_logger.close()

    def __del__(self) -> None:
        """
        Overview:
            Execute the close command and close the collector. __del__ is automatically called to \
                destroy the collector instance when the collector finishes its work
        """
        self.close()


if __name__ == '__main__':
    from ding.config.config import read_config_yaml
    from ding.policy.model_based.alphazero_policy import AlphaZeroPolicy
    from ding.envs import get_env_cls
    from ding.model import create_model

    cfg_path = os.path.join(os.getcwd(), 'alphazero_config_ding.yaml')
    cfg = read_config_yaml(cfg_path)

    env_fn = get_env_cls(cfg.env)
    collector_env = env_fn(cfg.env)
    model = create_model(cfg.model)
    policy = AlphaZeroPolicy(
        cfg.policy, model=model, enable_field=[
            'learn',
            'collect',
            'eval',
        ]
    )

    collector = AlphazeroCollector(
        cfg=cfg.policy.collect.collector,
        env=collector_env,
        policy=policy.collect_mode,
        # tb_logger=tb_logger,
        exp_name=cfg.exp_name
    )
    iter_count = 0
    while iter_count < 100:
        output = collector.collect()
        iter_count += 1
    # print(output)
