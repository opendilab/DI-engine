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
        Returns:
            - reward (:obj:`Any`)
        """
        raise NotImplementedError()

    @abstractmethod
    def train(self, data) -> None:
        raise NotImplementedError()

    @abstractmethod
    def load_expert_data(self, data) -> None:
        raise NotImplementedError()

    @abstractmethod
    def collect_data(self, data) -> None:
        raise NotImplementedError()

    @abstractmethod
    def clear_data(self) -> None:
        raise NotImplementedError()


def create_reward_model(cfg: dict, device: str, tb_logger: 'SummaryWriter') -> BaseRewardModel:  # noqa
    """
    Overview:
        estimate reward
    Arguments:
        - data (:obj:`List`): the list of data used for estimation
    Returns:
        - reward (:obj:`Any`)
    """
    cfg = copy.deepcopy(cfg)
    if 'import_names' in cfg:
        import_module(cfg.pop('import_names'))
    reward_model_type = cfg.pop('type')
    return REWARD_MODEL_REGISTRY.build(reward_model_type, cfg, device=device, tb_logger=tb_logger)
