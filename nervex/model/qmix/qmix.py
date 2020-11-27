import torch
import torch.nn as nn
import torch.nn.functional as F


class Mixer(nn.Module):
    """
    Overview:
        mixer network in QMIX, which mix up the independent q_value of each agent to a total q_value
    Interface:
        __init__, forward
    """

    def __init__(self, agent_num: int, embedding_dim: int, w_layers: int = 2) -> None:
        """
        Overview:
            initialize mixer network
        Arguments:
            - agent_num (:obj:`int`): the number of agent
            - embedding_dim (:obj:`int`): the dimension of state emdedding
            - w_layers (;obj:`int`): the number of fully-connected layers of mixer weight
        """
        super(Mixer, self).__init__()
        self._agent_num = agent_num
        self._embedding_dim = embedding_dim
        self._act = nn.ReLU()
        self._w1 = []
        for i in range(w_layers - 1):
            self._w1.append(nn.Linear(embedding_dim, embedding_dim))
            self._w1.append(self._act)
        self._w1.append(nn.Linear(embedding_dim, embedding_dim * agent_num))
        self._w1 = nn.Sequential(*self._w1)

        self._w2 = []
        for i in range(w_layers - 1):
            self._w2.append(nn.Linear(embedding_dim, embedding_dim))
            self._w2.append(self._act)
        self._w2.append(nn.Linear(embedding_dim, embedding_dim))
        self._w2 = nn.Sequential(*self._w2)

        self._b1 = nn.Linear(embedding_dim, embedding_dim)
        self._b2 = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), self._act, nn.Linear(embedding_dim, 1))

    def forward(self, agent_q: torch.Tensor, state_embedding: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            forward computation graph of mixer network
        Arguments:
            - agent_q (:obj:`torch.FloatTensor`): the independent q_value of each agent
            - state_embedding (:obj:`torch.FloatTensor`): the emdedding vector of global state
        Returns:
            - total_q (:obj:`torch.FloatTensor`): the total mixed q_value
        Shapes:
            - agent_q (:obj:`torch.FloatTensor`): :math:`(B, N)`, where B is batch size and N is agent_num
            - state_embedding (:obj:`torch.FloatTensor`): :math:`(B, M)`, where M is embedding_dim
            - total_q (:obj:`torch.FloatTensor`): :math:`(B, )`
        """
        agent_q = agent_q.reshape(-1, 1, self._agent_num)
        # first layer
        w1 = torch.abs(self._w1(state_embedding)).reshape(-1, self._agent_num, self._embedding_dim)
        b1 = self._b1(state_embedding).reshape(-1, 1, self._embedding_dim)
        hidden = F.elu(torch.bmm(agent_q, w1) + b1)  # bs, 1, embedding_dim
        # second layer
        w2 = torch.abs(self._w2(state_embedding)).reshape(-1, self._embedding_dim, 1)
        b2 = self._b2(state_embedding).reshape(-1, 1, 1)
        hidden = torch.bmm(hidden, w2) + b2
        return hidden.squeeze(-1).squeeze(-1)
