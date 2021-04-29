from abc import ABC, abstractmethod
import copy
from nervex.utils import REWARD_MODEL_REGISTRY, import_module


class BaseRewardModel(ABC):
    """
        the base class of reward model
    """

    @abstractmethod
    def estimate(self, data: list) -> None:
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


def create_irl_model(cfg: dict, device: str, tb_logger: 'SummaryWriter') -> BaseRewardModel:  # noqa
    cfg = copy.deepcopy(cfg)
    if 'import_names' in cfg:
        import_module(cfg.pop('import_names'))
    irl_model_type = cfg.pop('type')
    return REWARD_MODEL_REGISTRY.build(irl_model_type, cfg, device=device, tb_logger=tb_logger)
