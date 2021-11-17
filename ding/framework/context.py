from typing import Callable, Optional


class Context(dict):
    """
    Overview:
        Context is an object that pass contextual data between middlewares, whose life cycle
        is only one training iteration. It is a dict that reflect itself, so you can set
        any properties as you wish.
    """

    def __init__(self, total_step: int = 0, prev: Optional["Context"] = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        self.total_step = total_step  # Total steps
        self.prev = prev

        # Reserved properties
        self._finish = False
        self._hooks_after_renew = []
        self.keep("_finish")

    def renew(self) -> 'Context':  # noqa
        """
        Overview:
            Renew context from self, add total_step and shift kept properties to the new instance.
        """
        ctx = Context()
        for hook in self._hooks_after_renew:
            hook(ctx, self)
        return ctx

    def keep(self, *keys: str) -> None:
        """
        Overview:
            Keep this key/keys until next iteration.
        """

        def _keep(new_, old):
            for key in keys:
                new_[key] = old[key]

        self.after_renew(_keep)

    def after_renew(self, fn: Callable) -> None:
        """
        Overview:
            Hook after renew, the function should look like (lambda new_, old: ...)
        Arguments:
            - fn (:obj:`Callable`): Hook after renew
        """
        self._hooks_after_renew.append(fn)

    def finish(self, finish: bool = True) -> None:
        """
        Overview:
            Set finish flag on context
        Arguments:
            - finish (:obj:`bool`): Finish or not
        """
        self._finish = finish

    # Make it pickable
    def __getstate__(self):
        _ctx = {}
        for key, value in self.items():
            if key in ["_hooks_after_renew"]:
                continue
            _ctx[key] = value
        return _ctx

    def __setstate__(self, d):
        self.__dict__ = d

    def __reduce__(self):
        return (Context.new, (self.__getstate__(), ))

    @staticmethod
    def new(ctx):
        return Context(**ctx)
