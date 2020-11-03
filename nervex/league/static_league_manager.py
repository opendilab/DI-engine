import copy
from nervex.league import BaseLeagueManager, register_league
from nervex.league.player import ActivePlayer
from nervex.rl_utils import epsilon_greedy


class StaticLeagueManager(BaseLeagueManager):
    # override
    def _init_league(self):
        super()._init_league()
        self.exploration = epsilon_greedy(0.95, 0.05, self.cfg.exploration.decay_len)

    # override
    def _get_task_info(self, player):
        assert isinstance(player, ActivePlayer), player.__class__
        agent_step = player.total_agent_step
        eps = self.exploration(agent_step)
        env_num = self.cfg.task.env_num
        model_config = copy.deepcopy(self.model_config)
        task_info = {
            'episode_num': self.cfg.task.episode_num,
            'env_num': env_num,
            'agent_num': 1,
            'data_push_length': self.cfg.task.data_push_length,
            'agent_update_freq': self.cfg.task.agent_update_freq,
            'compressor': self.cfg.task.compressor,
            'forward_kwargs': {
                'eps': eps,
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
        return task_info

    # override
    def _mutate_player(self, player):
        pass

    # override
    def _update_player(self, player, player_info):
        if isinstance(player, ActivePlayer):
            train_step = player_info['train_step']
            player.total_agent_step = train_step


register_league('static', StaticLeagueManager)
