from abc import ABC, abstractmethod
import copy
from nervex.utils import REWARD_MODEL_REGISTRY, import_module


class BaseRewardModel(ABC):
    """
    Overview:
        the base class of reward model
    Interface:
        ``estimate``, ``train``, ``load_expert_data``, ``collect_data``, ``clear_date``
    """

    @abstractmethod
    def estimate(self, data: list) -> None:
        """
        Overview:
            estimate reward
        Arguments:
            - data (:obj:`List`): the list of data used for estimation
        Returns / Effects:
            - This can be a side effect function which updates the reward value
            - If this function returns, an example returned object can be reward (:obj:`Any`): the estimated reward
        """
        raise NotImplementedError()

    @abstractmethod
    def train(self, data) -> None:
        """
        Overview:
            Training the reward model
        Arguments:
            - data (:obj:`Any`): Data used for training
        Effects:
            - This is mostly a side effect function which updates the reward model
        """
        raise NotImplementedError()

    @abstractmethod
    def load_expert_data(self, data) -> None:
        """
        Overview:
            Getting the expert data
        Arguments:
            - data (:obj:`Any`): Expert data
        Effects:
            This is mostly a side effect function which updates the expert data attribute (e.g.  ``self.expert_data``)
        """
        raise NotImplementedError()

    @abstractmethod
    def collect_data(self, data) -> None:
        """
        Overview:
            Collecting training data in designated formate or with designated transition.
        Arguments:
            - data (:obj:`Any`): Raw training data (e.g. some form of states, actions, obs, etc)
        Returns / Effects:
            - This can be a side effect function which updates the data attribute in ``self``
        """
        raise NotImplementedError()

    @abstractmethod
    def clear_data(self) -> None:
        """
        Overview:
            Clearing training data. \
            This can be a side effect function which clears the data attribute in ``self``
        """
        raise NotImplementedError()


def create_reward_model(cfg: dict, device: str, tb_logger: 'SummaryWriter') -> BaseRewardModel:  # noqa
    """
    Overview:
        Reward Estimation Model.
    Arguments:
        - cfg (:obj:`Dict`): Training config
        - device (:obj:`str`): Device usage, i.e. "cpu" or "cuda"
        - tb_logger (:obj:`str`): Logger, defaultly set as 'SummaryWriter' for model summary
    Returns:
        - reward (:obj:`Any`): The reward model
    """
    cfg = copy.deepcopy(cfg)
    if 'import_names' in cfg:
        import_module(cfg.pop('import_names'))
    reward_model_type = cfg.pop('type')
    return REWARD_MODEL_REGISTRY.build(reward_model_type, cfg, device=device, tb_logger=tb_logger)
