from typing import Dict, List
import pickle
import random

import torch.optim as optim

from ding.utils import REWARD_MODEL_REGISTRY, one_time_warning
from .base_reward_model import BaseRewardModel
from .network import REDNetwork
from .reword_model_utils import concat_state_action_pairs


@REWARD_MODEL_REGISTRY.register('red')
class RedRewardModel(BaseRewardModel):
    """
    Overview:
         The implement of reward model in RED (https://arxiv.org/abs/1905.06750)
    Interface:
        ``estimate``, ``train``, ``load_expert_data``, ``collect_data``, ``clear_date``, \
            ``__init__``, ``_train``
    Config:
        == ==================  ======   =============  =======================================  =======================
        ID Symbol              Type     Default Value  Description                              Other(Shape)
        == ==================  ======   =============  =======================================  =======================
        1  ``type``             str      red           | Reward model register name, refer      |
                                                       | to registry ``REWARD_MODEL_REGISTRY``  |
        2  | ``expert_data_``   str      expert_data   | Path to the expert dataset             | Should be a '.pkl'
           | ``path``                    .pkl          |                                        | file
        3  | ``sample_size``    int      1000          | sample data from expert dataset        |
                                                       | with fixed size                        |
        4  | ``sigma``          int      5             | hyperparameter of r(s,a)               | r(s,a) = exp(
                                                                                                | -sigma* L(s,a))
        5  | ``batch_size``     int      64            | Training batch size                    |
        6  | ``hidden``         list     [64, 64,      | Sequence of ``hidden_size``            |
           | ``_size_list``     (int)    128]          | of reward network                      |
        7  | ``update_per_``    int      100           | Number of updates per collect          |
           | ``collect``                               |                                        |
        8  | ``clear_buffer``   int      1             | clear buffer per fixed iters           | make sure replay
             ``_per_iters``                                                                     | buffer's data count
                                                                                                | isn't too few.
                                                                                                | (code work in entry)
        == ==================  ======   =============  =======================================  =======================
    """
    config = dict(
        # (str) Reward model register name, refer to registry ``REWARD_MODEL_REGISTRY``.
        type='red',
        # (int) observation shape
        # obs_shape=4,
        # (int) action shape
        # action_shape=1,
        # (int) Sample data from expert dataset with fixed size.
        sample_size=1000,
        # (list(int)) Sequence of ``hidden_size`` of reward network.
        hidden_size_list=[128, 1],
        # (float) The step size of gradient descent.
        learning_rate=1e-3,
        # (int) How many updates(iterations) to train after collector's one collection.
        # Bigger "update_per_collect" means bigger off-policy.
        # collect data -> update policy-> collect data -> ...
        update_per_collect=100,
        # (str) Path to the expert dataset
        # expert_data_path='expert_data.pkl',
        # (int) How many samples in a training batch.
        batch_size=64,
        # (float) Hyperparameter at estimated score of r(s,a).
        # r(s,a) = exp(-sigma* L(s,a))
        sigma=0.5,
        # (int) Clear buffer per fixed iters.
        clear_buffer_per_iters=1,
    )

    def __init__(self, config: Dict, device: str, tb_logger: 'SummaryWriter') -> None:  # noqa
        """
        Overview:
            Initialize ``self.`` See ``help(type(self))`` for accurate signature.
        Arguments:
            - cfg (:obj:`Dict`): Training config
            - device (:obj:`str`): Device usage, i.e. "cpu" or "cuda"
            - tb_logger (:obj:`str`): Logger, defaultly set as 'SummaryWriter' for model summary
        """
        super(RedRewardModel, self).__init__()
        self.cfg: Dict = config
        self.expert_data: List[tuple] = []
        self.device = device
        assert device in ["cpu", "cuda"] or "cuda" in device
        self.tb_logger = tb_logger
        self.reward_model = REDNetwork(config.obs_shape, config.action_shape, config.hidden_size_list, config.sigma)
        self.reward_model.to(self.device)
        self.opt = optim.Adam(self.reward_model.predictor.parameters(), config.learning_rate)
        self.train_once_flag = False

        self.load_expert_data()

    def load_expert_data(self) -> None:
        """
        Overview:
            Getting the expert data from ``config['expert_data_path']`` attribute in self.
        Effects:
            This is a side effect function which updates the expert data attribute (e.g.  ``self.expert_data``)
        """
        with open(self.cfg.expert_data_path, 'rb') as f:
            self.expert_data = pickle.load(f)
        sample_size = min(len(self.expert_data), self.cfg.sample_size)
        self.expert_data = random.sample(self.expert_data, sample_size)
        print('the expert data size is:', len(self.expert_data))

    def _train(self) -> float:
        """
        Overview:
            Helper function for ``train`` which caclulates loss for train data and expert data.
        Returns:
            - Combined loss calculated of reward model from using ``states_actions_tensor``.
        """
        sample_batch = random.sample(self.expert_data, self.cfg.batch_size)
        states_actions_tensor = concat_state_action_pairs(sample_batch)
        states_actions_tensor = states_actions_tensor.to(self.device)
        loss = self.reward_model.learn(states_actions_tensor)
        # loss = F.mse_loss(predict_feature, target_feature.detach())
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()

    def train(self) -> None:
        """
        Overview:
            Training the RED reward model. In default, RED model should be trained once.
        Effects:
            - This is a side effect function which updates the reward model and increment the train iteration count.
        """
        if self.train_once_flag:
            one_time_warning('RED model should be trained once, we do not train it anymore')
        else:
            for i in range(self.cfg.update_per_collect):
                loss = self._train()
                self.tb_logger.add_scalar('reward_model/red_loss', loss, i)
            self.train_once_flag = True

    def estimate(self, data: list) -> List[Dict]:
        """
        Overview:
            Estimate reward by rewriting the reward key
        Arguments:
            - data (:obj:`list`): the list of data used for estimation, \
                with at least ``obs`` and ``action`` keys.
        Effects:
            - This is a side effect function which updates the reward values in place.
        """
        # NOTE: deepcopy reward part of data is very important,
        # otherwise the reward of data in the replay buffer will be incorrectly modified.
        train_data_augmented = self.reward_deepcopy(data)
        states_actions_tensor = concat_state_action_pairs(train_data_augmented)
        states_actions_tensor = states_actions_tensor.to(self.device)
        r = self.reward_model.forward(states_actions_tensor)
        for item, rew in zip(train_data_augmented, r):
            item['reward'] = rew
        return train_data_augmented

    def collect_data(self, data) -> None:
        """
        Overview:
            Collecting training data, not implemented if reward model (i.e. online_net) is only trained ones, \
                if online_net is trained continuously, there should be some implementations in collect_data method
        """
        # if online_net is trained continuously, there should be some implementations in collect_data method
        pass

    def clear_data(self, iter: int):
        """
        Overview:
            Collecting clearing data, not implemented if reward model (i.e. online_net) is only trained ones, \
                if online_net is trained continuously, there should be some implementations in clear_data method
        """
        # if online_net is trained continuously, there should be some implementations in clear_data method
        pass
