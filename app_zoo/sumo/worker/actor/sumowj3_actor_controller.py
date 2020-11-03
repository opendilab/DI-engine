from collections import namedtuple
from typing import List, Dict, Any

from nervex.model import FCDQN
from nervex.worker.actor import ZerglingActor, register_actor
from app_zoo.sumo.envs import SumoWJ3Env, FakeSumoWJ3Env
from app_zoo.sumo.worker.agent.sumo_dqn_agent import SumoDqnActorAgent


class SumoWJ3Actor(ZerglingActor):
    # override
    def _setup_env_fn(self, env_cfg: Dict) -> None:
        env_fn_mapping = {'normal': SumoWJ3Env, 'fake': FakeSumoWJ3Env}
        self._env_fn = env_fn_mapping[env_cfg.env_type]

    # override
    def _setup_agents(self) -> None:
        agent_cfg = self._job['agent']
        self._agent_name = list(agent_cfg.keys())[0]
        env_info = self._env_manager._env_ref.info()
        model = FCDQN(env_info.obs_space.shape, list(env_info.act_space.shape.values()))
        if self._cfg.actor.use_cuda:
            model.cuda()
        self._agent = SumoDqnActorAgent(model)

    # override
    def _get_transition(self, obs: Any, agent_output: Dict, timestep: namedtuple) -> Dict:
        data = {
            'obs': obs,
            'next_obs': timestep.obs,
            'q_value': agent_output['logits'],
            'action': agent_output['action'],
            'reward': timestep.reward,
            'done': timestep.done,
            'priority': 1.0,
        }
        return data

    # override
    def __repr__(self) -> str:
        return "SumoWJ3Actor"


register_actor('sumowj3', SumoWJ3Actor)
