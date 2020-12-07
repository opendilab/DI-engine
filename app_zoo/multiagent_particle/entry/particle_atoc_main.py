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

    def _cal_delta_Q(self, data):
        obs = data['obs']
        assert len(obs.shape) == 3
        batch_size = obs.shape[0]
        thought = data['thougths']
        old_thought = data['old_thoughts']
        C = data['groups']
        for b in range(batch_size):
            for i in range(self._n_agent):
                if not C[b][i][i]:
                    continue
                q_group = []
                actual_q_group = []
                for j in range(self._n_agent):
                    if not C[b][i][j]:
                        continue
                    before_update_action_j = self._actor_net.actor_2(old_thought[b][j])
                    after_update_action_j = self._actor_net.actor_2(thought[b][j])
                    before_update_Q_j = self._critic_net({
                        'obs': obs[b][j],
                        'action': before_update_action_j[b][j]
                    })['q_value']
                    after_update_Q_j = self._critic_net({
                        'obs': obs[b][j],
                        'action': after_update_action_j[b][j]
                    })['q_value']
                    q_group.append(before_update_Q_j)
                    actual_q_group.append(after_update_Q_j)
                q_group = torch.stack(q_group)
                actual_q_group = torch.stack(actual_q_group)
                delta_q = actual_q_group.mean() - q_group.mean()
                self._D.put((thought[b][i], delta_q))

    def _updata_attention_unit(self):
        thought_batch = []
        delta_q_batch = []
        while not self._D.empty():
            thought, delta_q = self._D.get()
            thought_batch.append(thought)
            delta_q_batch.append(delta_q)

        # shape (len, thougth_dim)
        thought_batch = torch.stack(thought_batch)

        # shape (len, 1)
        pi = self._actor_net.attention(thought_batch)
        # shape (len, 1)
        delta_q_batch = torch.stack(delta_q_batch)

        # When an episode ends, we perform min-max normalization on delta_Q in D and get delta_Q in [0, 1]
        # shape (len, 1)
        delta_Q = (delta_q_batch - delta_q_batch.min()) / (delta_q_batch.max() - delta_q_batch.min())

        loss = -delta_Q * torch.log(pi) - (1.0 - delta_Q) * torch.log(1.0 - pi)

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
