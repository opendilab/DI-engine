"""配置从内部 Dreamer 包导出，供脚本和模块使用。"""

if __package__:
    from .dreamer.config import TASKS, VERSIONS, SOLVED_RETURN, DreamerConfig, TrainConfig
else:
    from dreamer.config import TASKS, VERSIONS, SOLVED_RETURN, DreamerConfig, TrainConfig

__all__ = ["TASKS", "VERSIONS", "SOLVED_RETURN", "DreamerConfig", "TrainConfig"]
