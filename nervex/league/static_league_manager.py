from nervex.league import BaseLeagueManager, register_league


class StaticLeagueManager(BaseLeagueManager):
    # override
    def _get_task_info(self, player):
        env_num = self.cfg.task.env_num
        eps = 0.9
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
                    'model': {},
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
        pass


register_league('static', StaticLeagueManager)
