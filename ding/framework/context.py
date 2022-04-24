import numpy as np


class Context(dict):
    """
    Overview:
        Context is an object that pass contextual data between middlewares, whose life cycle
        is only one training iteration. It is a dict that reflect itself, so you can set
        any properties as you wish.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        self._kept_keys = set()
        self.total_step = 0

    def renew(self) -> 'Context':  # noqa
        """
        Overview:
            Renew context from self, add total_step and shift kept properties to the new instance.
        """
        total_step = self.total_step
        ctx = type(self)()
        for key in self._kept_keys:
            if key in self:
                ctx[key] = self[key]
        ctx.total_step = total_step + 1
        return ctx

    def keep(self, *keys: str) -> None:
        """
        Overview:
            Keep this key/keys until next iteration.
        """
        for key in keys:
            self._kept_keys.add(key)


class OnlineRLContext(Context):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        # common
        self.total_step = 0
        self.env_step = 0
        self.env_episode = 0
        self.train_iter = 0
        self.train_data = None
        # collect
        self.collect_kwargs = {}
        self.trajectories = None
        self.episodes = None
        self.trajectory_end_idx = []
        # eval
        self.eval_value = -np.inf
        self.last_eval_iter = -1

        self.keep('env_step', 'env_episode', 'train_iter', 'last_eval_iter')


class OfflineRLContext(Context):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        # common
        self.total_step = 0
        self.train_epoch = 0
        self.train_iter = 0
        self.train_data = None
        # eval
        self.eval_value = -np.inf
        self.last_eval_iter = -1

        self.keep('train_iter', 'last_eval_iter')
