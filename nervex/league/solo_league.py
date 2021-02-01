import copy
from easydict import EasyDict
import os.path as osp

from nervex.league import BaseLeague, register_league
from nervex.league.player import ActivePlayer
from nervex.utils import read_config, deep_merge_dicts
from nervex.config import solo_league_default_config


class SoloLeague(BaseLeague):
    """
    Overview:
        Solo game league. Solo game means only one player is permitted to interact with the game env,
        rather than two players fighting against each other.
        Unlike ``BattleLeague`` would decide which two players will play against,
        ``SoloLeague`` only focuses on how the player will interact with the game env, not another player.
    Interface:
        __init__, run, close, finish_job, update_active_player
    """

    # override
    def _init_cfg(self, cfg: EasyDict) -> None:
        cfg = deep_merge_dicts(solo_league_default_config, cfg)
        self.cfg = cfg.league
        self.model_config = cfg.get('model', EasyDict())

    # override
    def _get_job_info(self, player: ActivePlayer) -> dict:
        """
        Overview:
            Get player's job related info, called by ``_launch_job``.
        Arguments:
            - player (:obj:`ActivePlayer`): the active player that will be assigned a job
        """
        assert isinstance(player, ActivePlayer), player.__class__
        player_job_info = EasyDict(player.get_job())
        model_config = copy.deepcopy(self.model_config)
        job_info = {
            'env_kwargs': player_job_info.env_kwargs,
            'agent_num': 1,
            'agent_update_freq': player_job_info.agent_update_freq,
            'compressor': player_job_info.compressor,
            # 'forward_kwargs': player_job_info.forward_kwargs,
            # 'adder_kwargs': player_job_info.adder_kwargs,
            'launch_player': player.player_id,
            'player_id': [player.player_id],  # for solo game, it is a list with only one player_id
            'agent': {
                '0': {
                    'name': '0',
                    'model': model_config,
                    'agent_update_path': player.checkpoint_path,
                },
            },
        }
        return job_info

    # override
    def _mutate_player(self, player: ActivePlayer):
        """
        Overview:
            Players have the probability to be reset to supervised learning model parameters.
        Arguments:
            - player (:obj:`ActivePlayer`): the active player that may mutate
        """
        pass

    # override
    def _update_player(self, player: ActivePlayer, player_info: dict) -> None:
        """
        Overview:
            Update an active player, called by ``self.update_active_player``.
        Arguments:
            - player (:obj:`ActivePlayer`): the active player that will be updated
            - player_info (:obj:`dict`): an info dict of the active player which is to be updated
        """
        if isinstance(player, ActivePlayer):
            train_step = player_info['train_step']
            player.total_agent_step = train_step


register_league('solo', SoloLeague)
