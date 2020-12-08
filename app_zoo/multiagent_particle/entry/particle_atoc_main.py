import torch
from nervex.model.atoc import ATOCActorNet, ATOCCriticNet
from app_zoo.multiagent_particle.envs import ParicleEnv
import queue
from nervex.torch_utils import to_tensor, tensor_to_list


class ATOCTrainer:

    def __init__(self, obs_dim, action_dim, thought_dim, n_agent, m_group=5, t_initiate=15):
        self._n_agent = n_agent
        self._act_dim = action_dim
        self._obs_dim = obs_dim
        self._thought_dim = thought_dim
        self._actor_net = ATOCActorNet(obs_dim, thought_dim, action_dim, n_agent, m_group, t_initiate)
        self._critic_net = ATOCCriticNet(obs_dim, action_dim)
        # "ATOC paper: store (Qi, hi) into a queue D"
        self._D = queue.Queue()

    def step(self, obs):
        action = self._actor_net({'obs': obs})
        return action


def main():
    env = ParicleEnv({'env_name': "simple_spread", 'discrete_action': True})
    # init parameters according to env

    n_agent = env.agent_num

    # now ATOC only support same type agent
    obs_sp = env.info().obs_space.get('agent0')
    act_sp = env.info().act_space.get('agent0')
    action_dim = act_sp.value['max'] + 1 - act_sp.value['min']
    obs_dim = obs_sp.shape[0]

    trainer = ATOCTrainer(obs_dim, action_dim, 8, n_agent)
    obs_n = env.reset()
    # shape (A, obs_dim)
    for i in range(20):
        obs = torch.stack(obs_n)
        # unsqueeze to set batch dim
        # shape (1, A, obs_dim)
        action = trainer.step(obs.unsqueeze(0))['action'].squeeze(0)
        timestep = env.step(action)
        obs_n, rew_n, done_n, info_n = timestep
        print(timestep)


main()
