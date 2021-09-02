import os
import shutil
from easydict import EasyDict
from ding.league import BaseLeague, ActivePlayer


class DemoLeague(BaseLeague):

    def __init__(self, cfg):
        super(DemoLeague, self).__init__(cfg)
        self.reset_checkpoint_path = os.path.join(self.path_policy, 'reset_ckpt.pth')

    # override
    def _get_job_info(self, player: ActivePlayer, eval_flag: bool = False) -> dict:
        assert isinstance(player, ActivePlayer), player.__class__
        player_job_info = EasyDict(player.get_job(eval_flag))
        return {
            'agent_num': 2,
            'launch_player': player.player_id,
            'player_id': [player.player_id, player_job_info.opponent.player_id],
            'checkpoint_path': [player.checkpoint_path, player_job_info.opponent.checkpoint_path],
            'player_active_flag': [isinstance(p, ActivePlayer) for p in [player, player_job_info.opponent]],
        }

    # override
    def _mutate_player(self, player: ActivePlayer):
        for p in self.active_players:
            result = p.mutate({'reset_checkpoint_path': self.reset_checkpoint_path})
            if result is not None:
                p.rating = self.metric_env.create_rating()
                self.load_checkpoint(p.player_id, result)  # load_checkpoint is set by the caller of league
                self.save_checkpoint(result, p.checkpoint_path)

    # override
    def _update_player(self, player: ActivePlayer, player_info: dict) -> None:
        assert isinstance(player, ActivePlayer)
        if 'learner_step' in player_info:
            player.total_agent_step = player_info['learner_step']

    # override
    @staticmethod
    def save_checkpoint(src_checkpoint_path: str, dst_checkpoint_path: str) -> None:
        shutil.copy(src_checkpoint_path, dst_checkpoint_path)
