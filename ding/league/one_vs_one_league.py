from easydict import EasyDict
from typing import Optional

from ding.utils import LEAGUE_REGISTRY
from .base_league import BaseLeague
from .player import ActivePlayer


@LEAGUE_REGISTRY.register('one_vs_one')
class OneVsOneLeague(BaseLeague):
    """
    Overview:
        One vs One battle game league.
        Decide which two players will play against each other.
    Interface:
        __init__, run, close, finish_job, update_active_player
    """
    config = dict(
        league_type='one_vs_one',
        import_names=["ding.league"],
        # ---player----
        # "player_category" is just a name. Depends on the env.
        # For example, in StarCraft, this can be ['zerg', 'terran', 'protoss'].
        player_category=['default'],
        # Support different types of active players for solo and battle league.
        # For solo league, supports ['solo_active_player'].
        # For battle league, supports ['battle_active_player', 'main_player', 'main_exploiter', 'league_exploiter'].
        active_players=dict(
            naive_sp_player=1,  # {player_type: player_num}
        ),
        naive_sp_player=dict(
            # There should be keys ['one_phase_step', 'branch_probs', 'strong_win_rate'].
            # Specifically for 'main_exploiter' of StarCraft, there should be an additional key ['min_valid_win_rate'].
            one_phase_step=10,
            branch_probs=dict(
                pfsp=0.5,
                sp=0.5,
            ),
            strong_win_rate=0.7,
        ),
        # "use_pretrain" means whether to use pretrain model to initialize active player.
        use_pretrain=False,
        # "use_pretrain_init_historical" means whether to use pretrain model to initialize historical player.
        # "pretrain_checkpoint_path" is the pretrain checkpoint path used in "use_pretrain" and
        # "use_pretrain_init_historical". If both are False, "pretrain_checkpoint_path" can be omitted as well.
        # Otherwise, "pretrain_checkpoint_path" should list paths of all player categories.
        use_pretrain_init_historical=False,
        pretrain_checkpoint_path=dict(default='default_cate_pretrain.pth', ),
        # ---payoff---
        payoff=dict(
            # Supports ['battle']
            type='battle',
            decay=0.99,
            min_win_rate_games=8,
        ),
        metric=dict(
            mu=0,
            sigma=25 / 3,
            beta=25 / 3 / 2,
            tau=0.0,
            draw_probability=0.02,
        ),
    )

    # override
    def _get_job_info(self, player: ActivePlayer, eval_flag: bool = False) -> dict:
        """
        Overview:
            Get player's job related info, called by ``_launch_job``.
        Arguments:
            - player (:obj:`ActivePlayer`): The active player that will be assigned a job.
        """
        assert isinstance(player, ActivePlayer), player.__class__
        player_job_info = EasyDict(player.get_job(eval_flag))
        if eval_flag:
            return {
                'agent_num': 1,
                'launch_player': player.player_id,
                'player_id': [player.player_id],
                'checkpoint_path': [player.checkpoint_path],
                'player_active_flag': [isinstance(player, ActivePlayer)],
                'eval_opponent': player_job_info.opponent,
            }
        else:
            return {
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
    def _update_player(self, player: ActivePlayer, player_info: dict) -> Optional[bool]:
        """
        Overview:
            Update an active player, called by ``self.update_active_player``.
        Arguments:
            - player (:obj:`ActivePlayer`): The active player that will be updated.
            - player_info (:obj:`dict`): An info dict of the active player which is to be updated.
        Returns:
            - increment_eval_difficulty (:obj:`bool`): Only return this when evaluator calls this method. \
                Return True if difficulty is incremented; Otherwise return False (difficulty will not increment \
                when it is already the most difficult or evaluator loses)
        """
        assert isinstance(player, ActivePlayer)
        if 'train_iteration' in player_info:
            # Update info from learner
            player.total_agent_step = player_info['train_iteration']
            return False
        elif 'eval_win' in player_info:
            if player_info['eval_win']:
                # Update info from evaluator
                increment_eval_difficulty = player.increment_eval_difficulty()
                return increment_eval_difficulty
            else:
                return False
