from collections import namedtuple
from typing import List, Dict, Any
import torch

from nervex.model import ConvValueAC
from nervex.worker.actor import ZerglingActor, register_actor
from nervex.worker.agent import ACActorAgent
from app_zoo.atari.envs import AtariEnv


class AtariActor(ZerglingActor):
    # override
    def _setup_env_fn(self, env_cfg: Dict) -> None:
        self._env_fn = AtariEnv

    # override
    def _setup_agent(self) -> None:
        agent_cfg = self._job['agent']
        self._agent_name = list(agent_cfg.keys())[0]
        sumo_env = AtariEnv(self._cfg.env)
        env_info = sumo_env.info()
        model = ConvValueAC(env_info.obs_space.shape, env_info.act_space.shape, self._job['agent'][self._agent_name]['model']['embedding_dim'])
        if self._cfg.actor.use_cuda:
            model.cuda()
        self._agent = ACActorAgent(model)
        self._agent.mode(train=False)

    # override
    def _get_transition(self, obs: Any, agent_output: Dict, timestep: namedtuple) -> Dict:
        data = {
            'obs': obs,
            'next_obs': timestep.obs,
            'logit': agent_output['logit'],
            'action': agent_output['action'],
            'value': agent_output['value'],
            'reward': timestep.reward,
            'done': timestep.done,
            'priority': 1.0,
        }
        return data

    # override
    def __repr__(self) -> str:
        return "AtariActor"


register_actor('atari_ac', AtariActor)
