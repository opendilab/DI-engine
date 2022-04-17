from abc import ABC, abstractmethod
from easydict import EasyDict
import copy
from typing import Any
from ding.utils import REWARD_MODEL_REGISTRY, import_module


class BaseRewardModel(ABC):
    """
    Overview:
        the base class of reward model
    Interface:
        ``default_config``, ``estimate``, ``train``, ``clear_data``, ``collect_data``, ``load_expert_date``
    """

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    @abstractmethod
    def estimate(self, data: list) -> Any:
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

    def load_expert_data(self, data) -> None:
        """
        Overview:
            Getting the expert data, usually used in inverse RL reward model
        Arguments:
            - data (:obj:`Any`): Expert data
        Effects:
            This is mostly a side effect function which updates the expert data attribute (e.g.  ``self.expert_data``)
        """
        pass

    def reward_deepcopy(self, train_data) -> Any:
        """
        Overview:
            this method deepcopy reward part in train_data, and other parts keep shallow copy
            to avoid the reward part of train_data in the replay buffer be incorrectly modified.
        Arguments:
            - train_data (:obj:`List`): the List of train data in which the reward part will be operated by deepcopy.
        """
        train_data_reward_deepcopy = [
            {k: copy.deepcopy(v) if k == 'reward' else v
             for k, v in sample.items()} for sample in train_data
        ]
        return train_data_reward_deepcopy


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
    if hasattr(cfg, 'reward_model'):
        reward_model_type = cfg.reward_model.pop('type')
    else:
        reward_model_type = cfg.pop('type')
    return REWARD_MODEL_REGISTRY.build(reward_model_type, cfg, device=device, tb_logger=tb_logger)


def get_reward_model_cls(cfg: EasyDict) -> type:
    import_module(cfg.get('import_names', []))
    return REWARD_MODEL_REGISTRY.get(cfg.type)
