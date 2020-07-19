from sc2learner.envs.common import EnvElementRunner
from sc2learner.envs.env.base_env import BaseEnv
from .alphastar_obs import ScalarObs, SpatialObs, EntityObs
from ..action.alphastar_action import AlphaStarRawAction


class AlphaStarObsRunner(EnvElementRunner):
    # override
    def _init(self, cfg: dict) -> None:
        self._obs_scalar = ScalarObs(cfg.obs_scalar)
        self._obs_spatial = SpatialObs(cfg.obs_spatial)
        self._obs_entity = EntityObs(cfg.obs_entity)
        self._obs_stat_type = cfg.obs_stat_type
        self._ignore_camera = cfg.ignore_camera
        self._agent_num = cfg.agent_num
        self._core = self._obs_entity  # placeholder
        self._map_size = cfg.map_size

    # override
    def reset(self) -> None:
        last_action = AlphaStarRawAction.Action(0, 0, None, None, None, None)
        self._last_action = [last_action for _ in range(self._agent_num)]
        self._repeat_action_type = [-1] * self._agent_num
        self._repeat_count = [0] * self._agent_num

    # override
    def get(self, engine: BaseEnv) -> dict:
        raw_obs = engine.raw_obs
        assert len(raw_obs) == self._agent_num
        obs = [None] * self._agent_num
        for i, o in enumerate(raw_obs):
            if o is not None:
                last_action = self._last_action[i]
                last_action = {k: getattr(last_action, k) for k in last_action._fields}
                last_action_type = last_action['action_type']
                if last_action_type == self._repeat_action_type[i]:
                    self._repeat_count[i] += 1
                else:
                    self._repeat_count[i] = 0
                    self._repeat_action_type[i] = last_action_type
                last_action['repeat_count'] = self._repeat_count[i]
                # merge last action
                o['last_action'] = last_action
                # merge stat
                o = self._merge_stat2obs(o, engine, i)
                # transform obs
                entity_info, entity_raw = self._obs_entity._to_agent_processor(o)
                obs[i] = {
                    'scalar_info': self._obs_scalar._to_agent_processor(o),
                    'spatial_info': self._obs_spatial._to_agent_processor(o),
                    'entity_info': entity_info,
                    'entity_raw': entity_raw,
                    'map_size': [self._map_size[1], self._map_size[0]],  # x,y -> y,x
                }
                obs[i] = self._mask_obs(obs[i])
        return obs

    def update_last_action(self, engine: BaseEnv) -> None:
        action = engine.action
        for n in range(self._agent_num):
            # only valid action update these variable
            if action[n] is not None:
                self._last_action[n] = action[n]

    # override
    def __repr__(self) -> str:
        return 'scalar: {}\tspatial: {}\tentity: {}'.format(
            repr(self._obs_scalar), repr(self._obs_spatial), repr(self._obs_entity)
        )

    # override
    @property
    def info(self) -> dict:
        return {'scalar': self._obs_scalar.info, 'spatial': self._obs_spatial.info, 'entity': self._obs_entity.info}

    def _mask_obs(self, obs: dict) -> dict:
        if self._ignore_camera:
            obs['spatial_info'][1:3] *= 0
            obs['entity_info'][:, 408:410] *= 0
        return obs

    def _merge_stat2obs(self, obs, engine, agent_no, game_loop=None):
        assert engine.loaded_eval_stat[agent_no] is not None, "please call load_stat method first"
        if self._obs_stat_type == 'replay_online':
            stat = engine.loaded_eval_stat[agent_no].get_input_z_by_game_loop(game_loop=game_loop)
        elif self._obs_stat_type == 'self_online':
            cumulative_stat = engine.episode_stat[agent_no].cumulative_statistics
            stat = engine.loaded_eval_stat[agent_no].get_input_z_by_game_loop(
                game_loop=None, cumulative_stat=cumulative_stat
            )
        elif self._obs_stat_type == 'replay_last':
            stat = engine.loaded_eval_stat[agent_no].get_input_z_by_game_loop(game_loop=None)
        else:
            raise ValueError(f"{self._obs_stat_type} unknown!")

        assert set(stat.keys()) == {'mmr', 'beginning_build_order', 'cumulative_stat'}
        obs.update(stat)
        return obs
