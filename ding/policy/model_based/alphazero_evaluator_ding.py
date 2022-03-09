import os
from collections import namedtuple
from ding.envs import BaseEnv
from typing import Optional, Any, List
from easydict import EasyDict
import numpy as np
from ding.utils.data import default_decollate
from ding.utils import build_logger, EasyTimer, SERIAL_COLLECTOR_REGISTRY


@SERIAL_COLLECTOR_REGISTRY.register('alphazero_eval')
class AlphazeroEvaluator:

    def __init__(
        self,
        cfg: EasyDict,
        env: BaseEnv = None,
        policy: namedtuple = None,
        tb_logger: 'SummaryWriter' = None,  # noqa
        exp_name: Optional[str] = 'default_experiment',
        instance_name: Optional[str] = 'evaluator'
    ):
        self._cfg = cfg
        self._env = env
        self._policy = policy
        self._exp_name = exp_name
        self._instance_name = instance_name
        # self._deepcopy_obs = self._cfg.deepcopy_obs
        # self._transform_obs = self._cfg.transform_obs
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
        self._last_eval_iter = 0
        self._eval_freq = self._cfg.eval_freq
        self._default_n_episode = self._cfg.n_episode
        self._stop_value = self._cfg.stop_value
        self._eval_freq = self._cfg.eval_freq
        self._end_flag = False

    def should_eval(self, train_iter: int) -> bool:
        """
        Overview:
            Determine whether you need to start the evaluation mode, if the number of training has reached\
                the maximum number of times to start the evaluator, return True
        """
        if (train_iter - self._last_eval_iter) < self._eval_freq and train_iter != 0:
            return False
        self._last_eval_iter = train_iter
        return True

    def eval(self, test_episode_num=None):
        test_episode_num = test_episode_num if test_episode_num else self._default_n_episode

        sp_stop, sp_winrate = self.self_play((test_episode_num))
        expert_winrate_list_first, expert_winrate_list_next = self.start_play_with_expert(test_episode_num)
        expert_winrate_first = expert_winrate_list_first.mean()
        expert_stop = expert_winrate_first >= self._stop_value
        return sp_stop, sp_winrate

    def self_play(
        self,
        test_episode_num=1,
    ):
        winner_list = []

        for i in range(test_episode_num):
            self.env.reset()
            done = False
            while not done:
                action, move_probs = self._policy.forward(self.env)
                self.env.do_action(action)
                done, winner = self.env.game_end()
                if done:
                    winner_list.append(winner)
                    # if winner != -1:
                    #     print("Game end. Winner is player:", winner)
                    # else:
                    #     print("Game end. Tie")
                    break
                action = self.env.expert_action()
                self.env.do_action(action)
                done, winner = self.env.game_end()
                if done:
                    winner_list.append(winner)
                    # if winner != -1:
                    #     print("Game end. Winner is player:", winner)
                    # else:
                    #     print("Game end. Tie")
                    break

        winrate = ((np.array(winner_list) == 1) + 0.5 * (np.array(winner_list) == -1)).mean()
        self._logger.info(f'[EVALUATOR] self play winrate_list:{winner_list}')
        self._logger.info(f'[EVALUATOR] self play player1 winrate:{winrate}')
        self._logger.info(f'[EVALUATOR] self play player2 winrate:{1 - winrate}')
        stop = winrate >= self._stop_value
        return stop, winrate

    def start_play_with_expert(
        self,
        test_episode_num=1,
    ):
        winner_list = []
        print('agent start first')
        for i in range(test_episode_num):
            self.env.reset()
            done = False
            while not done:
                action, move_probs = self._policy.forward(self.env)
                self.env.do_action(action)
                done, winner = self.env.game_end()
                if done:
                    winner_list.append(winner)
                    # if winner != -1:
                    #     print("Game end. Winner is player:", winner)
                    # else:
                    #     print("Game end. Tie")
                    break
                action = self.env.expert_action()
                self.env.do_action(action)
                done, winner = self.env.game_end()
                if done:
                    winner_list.append(winner)
                    # if winner != -1:
                    #     print("Game end. Winner is player:", winner)
                    # else:
                    #     print("Game end. Tie")
                    break
        print(f'agent start first winner_list:{winner_list}')
        winrate_list_first = ((np.array(winner_list) == 1) + (np.array(winner_list) == -1) * 0.5)
        winner_list.clear()
        print('build-in bot start first')
        for i in range(test_episode_num):
            self.env.reset()
            done = False
            while not done:
                action = self.env.expert_action()
                self.env.do_action(action)
                done, winner = self.env.game_end()
                if done:
                    winner_list.append(winner)
                    # if winner != -1:
                    #     print("Game end. Winner is player:", winner)
                    # else:
                    #     print("Game end. Tie")
                    break

                action, move_probs = self._policy.forward(self.env)
                self.env.do_action(action)
                done, winner = self.env.game_end()
                if done:
                    winner_list.append(winner)
                    # if winner != -1:
                    #     print("Game end. Winner is player:", winner)
                    # else:
                    #     print("Game end. Tie")
                    break
        print(f'build-in bot start first winner_list:{winner_list}')
        winrate_list_next = ((np.array(winner_list) == 2) + (np.array(winner_list) == -1) * 0.5)

        return winrate_list_first, winrate_list_next

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

    evaluator = AlphazeroEvaluator(
        cfg=cfg.policy.eval.evaluator,
        env=collector_env,
        policy=policy.eval_mode,
        # tb_logger=tb_logger,
        exp_name=cfg.exp_name
    )
    output = evaluator.eval()
