import copy
from easydict import EasyDict
import os.path as osp

from nervex.utils import deep_merge_dicts
from nervex.config import one_vs_one_league_default_config
from .base_league import BaseLeague, register_league
from .player import ActivePlayer


class OneVsOneLeague(BaseLeague):
    """
    Overview:
        One vs One battle game league.
        Decide which two players will play against each other.
    Interface:
        __init__, run, close, finish_job, update_active_player
    """

    # override
    def _init_cfg(self, cfg: EasyDict) -> None:
        cfg = deep_merge_dicts(one_vs_one_league_default_config.league, cfg)
        self.cfg = cfg
        # self.model_config = cfg.get('model', EasyDict())

    # override
    def _get_job_info(self, player: ActivePlayer, eval_flag: bool = False) -> dict:
        """
        Overview:
            Get player's job related info, called by ``_launch_job``.
        Arguments:
            - player (:obj:`ActivePlayer`): The active player that will be assigned a job.
        """
        assert isinstance(player, ActivePlayer), player.__class__
        # model_config = copy.deepcopy(self.model_config)
        if eval_flag:
            return {
                # 'env_kwargs': player_job_info.env_kwargs,
                'agent_num': 1,
                'launch_player': player.player_id,
                'player_id': [player.player_id],
                'checkpoint_path': [player.checkpoint_path],
                'player_active_flag': [isinstance(player, ActivePlayer)],
            }
        else:
            player_job_info = EasyDict(player.get_job(eval_flag))
            return {
                # 'env_kwargs': player_job_info.env_kwargs,
                'agent_num': 2,
                'launch_player': player.player_id,
                'player_id': [player.player_id, player_job_info.opponent.player_id],
                'checkpoint_path': [player.checkpoint_path, player_job_info.opponent.checkpoint_path],
                'player_active_flag': [isinstance(p, ActivePlayer) for p in [player, player_job_info.opponent]],
            }

    # override
    def _mutate_player(self, player: ActivePlayer):
        """
        Overview:
            Players have the probability to be reset to supervised learning model parameters.
        Arguments:
            - player (:obj:`ActivePlayer`): The active player that may mutate.
        """
        pass

    # override
    def _update_player(self, player: ActivePlayer, player_info: dict) -> None:
        """
        Overview:
            Update an active player, called by ``self.update_active_player``.
        Arguments:
            - player (:obj:`ActivePlayer`): The active player that will be updated.
            - player_info (:obj:`dict`): An info dict of the active player which is to be updated.
        """
        if isinstance(player, ActivePlayer):
            if 'train_step' in player_info:
                player.total_agent_step = player_info['train_step']
                player.checkpoint_path = player_info['checkpoint_path']
            elif 'eval_win' in player_info and player_info['eval_win']:
                increment_eval_difficulty = player.increment_eval_difficulty()


register_league('one_vs_one', OneVsOneLeague)
