import copy
import queue
import time
from collections import namedtuple
from threading import Thread
from typing import List, Dict, Any

import torch

from nervex.worker.actor import ZerglingActor, register_actor
from app_zoo.gfootball.model.iql.iql_network import FootballIQL
from app_zoo.gfootball.envs import GfootballEnv
from app_zoo.gfootball.worker.agent.gfootball_agent import GfootballIqlActorAgent


class GfootballActor(ZerglingActor):
    # override
    def _setup_env_fn(self, env_cfg: Dict) -> None:
        self._env_fn = GfootballEnv

    # override
    def _setup_agent(self) -> None:
        agent_cfg = self._job['agent']
        self._agent_name = list(agent_cfg.keys())[0]
        model = FootballIQL(agent_cfg[self._agent_name]['model'])
        if self._cfg.actor.use_cuda:
            model.cuda()
        self._agent = GfootballIqlActorAgent(model)
        self._agent.mode(train=False)

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
        return 'GfootballActor({})'.format(self._actor_uid)


register_actor("gfootball", GfootballActor)
