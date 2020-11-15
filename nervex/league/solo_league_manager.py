import copy
from nervex.league import BaseLeagueManager, register_league
from nervex.league.player import ActivePlayer
from nervex.rl_utils import epsilon_greedy


class SoloLeagueManager(BaseLeagueManager):
    """
    Overview:
        Solo game league manager, has only one active player and it only interacts with the game env.
        Unlike ``BattleLeagueManager`` would decide who the player will play aginst,
        ``SoloLeagueManager`` only focus on how the player will interact with the game env, not another player.
    Interface:
        __init__, run, close, finish_job, update_active_player
    """
    # override
    def _init_league(self):
        super(SoloLeagueManager, self)._init_league()
        self.exploration = epsilon_greedy(0.95, 0.05, self.cfg.exploration.decay_len)

    # override
    def _get_job_info(self, player: ActivePlayer) -> dict:
        """
        Overview:
            Get player's job related info, called by ``_launch_job``.
        Arguments:
            - player (:obj:`ActivePlayer`): the active player that will be assigned a job
        """
        assert isinstance(player, ActivePlayer), player.__class__
        player_job_info = player.get_job(self.exploration)
        env_num = self.cfg.job.env_num
        model_config = copy.deepcopy(self.model_config)
        job_info = {
            'episode_num': self.cfg.job.episode_num,
            'env_num': env_num,
            'some_env_related_info': 'placeholder',  # TODO: what info can be passed? (game mode, scenario, difficulty)
            'agent_num': 1,
            'data_push_length': self.cfg.job.data_push_length,
            'agent_update_freq': self.cfg.job.agent_update_freq,
            'compressor': self.cfg.job.compressor,
            'forward_kwargs': {
                'eps': player_job_info['eps'],
            },
            'launch_player': player.player_id,
            'player_id': [player.player_id],
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


register_league('static', SoloLeagueManager)
