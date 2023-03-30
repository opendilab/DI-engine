import copy
from easydict import EasyDict
import pickle

from ding.utils import REWARD_MODEL_REGISTRY

from .trex_reward_model import TrexRewardModel


@REWARD_MODEL_REGISTRY.register('drex')
class DrexRewardModel(TrexRewardModel):
    """
    Overview:
        The Drex reward model class (https://arxiv.org/pdf/1907.03976.pdf)
    Interface:
        ``estimate``, ``train``, ``load_expert_data``, ``collect_data``, ``clear_date``, \
            ``__init__``, ``_train``,
    Config:
        == ====================  ======   =============  =======================================  ===============
        ID Symbol                Type       Default Value  Description                            Other(Shape)
        == ====================  ======   =============  =======================================  ===============
        1  ``type``              str       drex          | Reward model register name, refer      |
                                                         | to registry ``REWARD_MODEL_REGISTRY``  |
        3  | ``learning_rate``   float     0.00001       | learning rate for optimizer            |
        4  | ``update_per_``     int       100           | Number of updates per collect          |
           | ``collect``                                 |                                        |
        5  | ``batch_size``      int       64            | How many samples in a training batch   |
        6  | ``hidden_size``     int       128           | Linear model hidden size               |
        7  | ``num_trajs``       int       0             | Number of downsampled full             |
                                                         | trajectories                           |
        8  | ``num_snippets``    int       6000          | Number of short subtrajectories        |
                                                         | to sample                              |
        == ====================  ======   =============  =======================================  ================
    """
    config = dict(
        # (str) Reward model register name, refer to registry ``REWARD_MODEL_REGISTRY``.
        type='drex',
        # (float) The step size of gradient descent.
        learning_rate=1e-5,
        # (int) How many updates(iterations) to train after collector's one collection.
        # Bigger "update_per_collect" means bigger off-policy.
        # collect data -> update policy-> collect data -> ...
        update_per_collect=100,
        # (int) How many samples in a training batch.
        batch_size=64,
        # (int) Linear model hidden size
        hidden_size=128,
        # (int) Number of downsampled full trajectories.
        num_trajs=0,
        # (int) Number of short subtrajectories to sample.
        num_snippets=6000,
    )

    bc_cfg = None

    def __init__(self, config: EasyDict, device: str, tb_logger: 'SummaryWriter') -> None:  # noqa
        """
        Overview:
            Initialize ``self.`` See ``help(type(self))`` for accurate signature.
        Arguments:
            - cfg (:obj:`EasyDict`): Training config
            - device (:obj:`str`): Device usage, i.e. "cpu" or "cuda"
            - tb_logger (:obj:`SummaryWriter`): Logger, defaultly set as 'SummaryWriter' for model summary
        """
        super(DrexRewardModel, self).__init__(copy.deepcopy(config), device, tb_logger)

        self.demo_data = []
        self.load_expert_data()

    def load_expert_data(self) -> None:
        """
        Overview:
            Getting the expert data from ``config.expert_data_path`` attribute in self
        Effects:
            This is a side effect function which updates the expert data attribute \
                (i.e. ``self.expert_data``) with ``fn:concat_state_action_pairs``
        """
        super(DrexRewardModel, self).load_expert_data()

        with open(self.cfg.reward_model.offline_data_path + '/suboptimal_data.pkl', 'rb') as f:
            self.demo_data = pickle.load(f)

    def train(self):
        self._train()
        return_dict = self.pred_data(self.demo_data)
        res, pred_returns = return_dict['real'], return_dict['pred']
        self._logger.info("real: " + str(res))
        self._logger.info("pred: " + str(pred_returns))

        info = {
            "min_snippet_length": self.min_snippet_length,
            "max_snippet_length": self.max_snippet_length,
            "len_num_training_obs": len(self.training_obs),
            "lem_num_labels": len(self.training_labels),
            "accuracy": self.calc_accuracy(self.reward_model, self.training_obs, self.training_labels),
        }
        self._logger.info(
            "accuracy and comparison:\n{}".format('\n'.join(['{}: {}'.format(k, v) for k, v in info.items()]))
        )
