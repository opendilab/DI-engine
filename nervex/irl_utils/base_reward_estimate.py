from abc import ABC, abstractmethod


class BaseRewardModel(ABC):
    """
        the base class of reward model
    """

    @abstractmethod
    def estimate(self, s, a) -> float:
        raise NotImplementedError()

    @abstractmethod
    def train(self, data) -> None:
        raise NotImplementedError()

    @abstractmethod
    def load_expert_data(self, data) -> None:
        raise NotImplementedError()

    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def collect_data(self, data) -> None:
        raise NotImplementedError()

    @abstractmethod
    def clear_data(self) -> None:
        raise NotImplementedError()


irl_model_mapping = {}


def create_irl_model(cfg: dict, device: str) -> BaseRewardModel:
    irl_model_type = cfg.type
    if irl_model_type not in irl_model_mapping:
        raise KeyError("not support irl model type: {}".format(irl_model_type))
    else:
        return irl_model_mapping[irl_model_type](cfg, device)


def register_irl_model(name: str, irl_model: type) -> None:
    assert isinstance(name, str)
    assert issubclass(irl_model, BaseRewardModel)
    irl_model_mapping[name] = irl_model
