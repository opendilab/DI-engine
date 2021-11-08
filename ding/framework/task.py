from types import GeneratorType
from typing import Callable
from ding.framework import Context


class Task:
    """
    Tash will manage the execution order of the entire pipeline, register new middleware,
    and generate new context objects.
    """

    def __init__(self) -> None:
        self.middleware = []
        self.ctx = Context()

    def use(self, fn: Callable) -> None:
        """
        Overview:
            Register middleware to task. The middleware will be executed by it's registry order.
        Arguments:
            - fn (:obj:`Callable`): A middleware is a function with only one argument: ctx.
        """
        self.middleware.append(fn)

    def run(self, max_step: int = 1e10) -> None:
        """
        Overview:
            Execute the iterations, when reach the max_step or task.finish is true,
            The loop will be break.
        Arguments:
            - max_step (:obj:`int`): Max step of iterations.
        """
        if len(self.middleware) == 0:
            return
        for _ in range(max_step):
            for fn in self.middleware:
                self.forward(fn)
            self.backward()
            if self.finish:
                break
            self.renew()

    def forward(self, fn: Callable) -> None:
        """
        Overview:
            This function will execute the middleware until the first yield statment,
            or the end of the middleware.
        Arguments:
            - fn (:obj:`Callable`): Function with contain the ctx argument in middleware.
        """
        # TODO how to treat multiple yield
        # TODO how to use return value or send value to generator
        g = fn(self.ctx)
        if isinstance(g, GeneratorType):
            next(g)
            self.ctx._backward_stack.append(g)

    def backward(self) -> None:
        """
        Overview:
            Execute the rest part of middleware, by the reversed order of registry.
        """
        for g in reversed(self.ctx._backward_stack):
            for _ in g:
                pass

    def renew(self) -> None:
        """
        Overview:
            Renew the context instance, this function should be called after backward in the end of iteration.
        """
        self.ctx = self.ctx.renew()

    @property
    def finish(self) -> bool:
        """
        Overview:
            Link the ctx's finish state, in order to be easily called externally.
        """
        return self.ctx._finish
