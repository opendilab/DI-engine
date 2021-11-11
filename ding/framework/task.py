import concurrent.futures
from types import GeneratorType
from typing import Any, Awaitable, Callable, List, Union
from ding.framework import Context
import asyncio


def enable_async(func: Callable) -> Callable:
    """
    Overview:
        Empower the function with async ability.
    Arguments:
        - func (:obj:`Callable`): The original function.
    Returns:
        - runtime_handler (:obj:`Callable`): The wrap function.
    """

    def runtime_handler(task, *args, **kwargs) -> Union[Any, Awaitable]:
        """
        Overview:
            If task's async mode is enabled, execute the step in current loop executor asyncly,
            or execute the task sync.
        Arguments:
            - task (:obj:`Task`): The task instance.
        Returns:
            - result (:obj:`Union[Any, Awaitable]`): The result or future object of middleware.
        """
        if "async_mode" in kwargs:
            async_mode = kwargs.pop("async_mode")
        else:
            async_mode = task.async_mode
        if async_mode:
            # with concurrent.futures.ProcessPoolExecutor() as pool:
            t = task._loop.run_in_executor(None, func, task, *args, **kwargs)
            task._async_stack.append(t)
            return t
        else:
            return func(task, *args, **kwargs)

    return runtime_handler


class Task:
    """
    Tash will manage the execution order of the entire pipeline, register new middleware,
    and generate new context objects.
    """

    def __init__(self, async_mode: bool = False) -> None:
        self.middleware = []
        self.ctx = Context()
        self._backward_stack = []

        # Async workarounds
        self.async_mode = async_mode
        self._async_stack = []
        self._loop = None
        if async_mode:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()

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

    @enable_async
    def forward(self, fn: Callable, ctx: Context = None, backward_stack: List = None) -> None:
        """
        Overview:
            This function will execute the middleware until the first yield statment,
            or the end of the middleware.
        Arguments:
            - fn (:obj:`Callable`): Function with contain the ctx argument in middleware.
        """
        # TODO how to treat multiple yield
        # TODO how to use return value or send value to generator
        stack = backward_stack or self._backward_stack
        ctx = ctx or self.ctx
        g = fn(ctx)
        if isinstance(g, GeneratorType):
            next(g)
            stack.append(g)

    @enable_async
    def backward(self, backward_stack: List = None) -> None:
        """
        Overview:
            Execute the rest part of middleware, by the reversed order of registry.
        """
        stack = backward_stack or self._backward_stack
        while stack:
            # FILO
            g = stack.pop()
            try:
                next(g)
            except StopIteration:
                continue

    def sequence(self, *fns: List[Callable]) -> Callable:
        """
        Overview:
            Wrap functions and keep them run in sequence, Usually in order to avoid the confusion
            of dependencies in async mode.
        Arguments:
            - fn (:obj:`Callable`): Chain a sequence of middleware, wrap them into one middleware function.
        """

        def _sequence(ctx):
            backward_stack = []
            for fn in fns:
                self.forward(fn, ctx=ctx, backward_stack=backward_stack, async_mode=False)
            yield
            self.backward(backward_stack=backward_stack, async_mode=False)

        return _sequence

    def renew(self) -> None:
        """
        Overview:
            Renew the context instance, this function should be called after backward in the end of iteration.
        """
        self.ctx = self.ctx.renew()
        self._backward_stack = []
        if self._loop:
            # Blocking
            self._loop.run_until_complete(self.async_renew())

    async def async_renew(self) -> Awaitable[None]:
        while self._async_stack:
            # FIFO
            t = self._async_stack.pop(0)
            await t

    @property
    def finish(self) -> bool:
        """
        Overview:
            Link the ctx's finish state, in order to be easily called externally.
        """
        return self.ctx._finish
