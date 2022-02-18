class Context(dict):
    """
    Overview:
        Context is an object that pass contextual data between middlewares, whose life cycle
        is only one training iteration. It is a dict that reflect itself, so you can set
        any properties as you wish.
    """

    def __init__(self, total_step: int = 0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        self.total_step = total_step
        self._kept_keys = set()

    def renew(self) -> 'Context':  # noqa
        """
        Overview:
            Renew context from self, add total_step and shift kept properties to the new instance.
        """
        ctx = Context()
        for key in self._kept_keys:
            if key in self:
                ctx[key] = self[key]
        return ctx

    def keep(self, *keys: str) -> None:
        """
        Overview:
            Keep this key/keys until next iteration.
        """
        for key in keys:
            self._kept_keys.add(key)
