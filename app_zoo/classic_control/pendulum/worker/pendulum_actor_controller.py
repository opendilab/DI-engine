from collections import namedtuple
from typing import List, Dict, Any
import copy

from nervex.model import FCQAC
from nervex.worker.actor import ZerglingActor, register_actor
from nervex.worker.agent import create_qac_actor_agent
from app_zoo.classic_control.pendulum.envs import PendulumEnv
from nervex.torch_utils import to_device


class PendulumActor(ZerglingActor):
    # override
    def _setup_env_fn(self, env_cfg: Dict) -> None:
        env_fn_mapping = {'normal': PendulumEnv}
        self._env_fn = env_fn_mapping[env_cfg['env_type']]

    # override
    def _setup_agent(self) -> None:
        agent_cfg = self._job['agent']
        self._agent_name = list(agent_cfg.keys())[0]
        pendulum_env = PendulumEnv(self._env_kwargs['env_cfg'])
        env_info = pendulum_env.info()
        # continuous action space
        model = FCQAC(env_info.obs_space.shape, len(env_info.act_space.shape), env_info.act_space.value)
        if self._cfg.actor.use_cuda:
            model.cuda()
        self._agent = create_qac_actor_agent(model)
        self._agent.mode(train=False)
    
    # override
    def _agent_inference(self, obs: Dict[int, Any]) -> Dict[int, Any]:
        # save in obs_pool
        for k, v in obs.items():
            self._obs_pool[k] = copy.deepcopy(v)

        env_id = obs.keys()
        obs = self._collate_fn(list(obs.values()))
        if self._cfg.actor.use_cuda:
            obs = to_device(obs, 'cuda')
        data = self._agent.forward(obs, mode='compute_action_q', **self._job['forward_kwargs'])  # add 'mode' kwarg
        if self._cfg.actor.use_cuda:
            data = to_device(data, 'cpu')
        data = self._decollate_fn(data)
        data = {i: d for i, d in zip(env_id, data)}
        return data

    # override
    def _get_transition(self, obs: Any, agent_output: Dict, timestep: namedtuple) -> Dict:
        data = {
            'obs': obs,
            'next_obs': timestep.obs,
            'q_value': agent_output['q'],
            'action': agent_output['action'],
            'reward': timestep.reward,
            'done': timestep.done,
            'priority': 1.0,
        }
        return data

    # override
    def __repr__(self) -> str:
        return "PendulumActor"


register_actor('pendulum', PendulumActor)
