from collections import defaultdict
import logging
import time
import asyncio
import concurrent.futures
from types import GeneratorType
from typing import Awaitable, Callable, Dict, Generator, List, Optional, Union
from ding.framework.context import Context
from ding.framework.parallel import Parallel


def enable_async(func: Callable) -> Callable:
    """
    Overview:
        Empower the function with async ability.
    Arguments:
        - func (:obj:`Callable`): The original function.
    Returns:
        - runtime_handler (:obj:`Callable`): The wrap function.
    """

    def runtime_handler(task: "Task", *args, **kwargs) -> "Task":
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
            t = task._loop.run_in_executor(task._thread_pool, func, task, *args, **kwargs)
            task._async_stack.append(t)
            return task
        else:
            return func(task, *args, **kwargs)

    return runtime_handler


class Task:
    """
    Tash will manage the execution order of the entire pipeline, register new middleware,
    and generate new context objects.
    """

    def __init__(
            self,
            async_mode: bool = False,
            n_async_workers: int = 1,
            middleware: List[Callable] = None,
            event_listeners: Dict[str, List] = None,
            once_listeners: Dict[str, List] = None,
            attach_callback: Callable = None,
            *args,
            **kwargs
    ) -> None:
        self.middleware = middleware or []
        self.ctx = Context()
        self._backward_stack = []

        # Async segment
        self.async_mode = async_mode
        self.n_async_workers = n_async_workers
        self._async_stack = []
        self._loop = None
        self._thread_pool = None
        self.event_listeners = event_listeners or defaultdict(list)
        self.once_listeners = once_listeners or defaultdict(list)

        # Parallel segment
        self.router = Parallel()
        if async_mode or self.router.is_active:
            self._thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=n_async_workers)
            self._loop = asyncio.new_event_loop()

        if self.router.is_active:
            self.router.register_rpc("task.emit", self.emit)
            if attach_callback:
                self.wait_for_attach_callback(attach_callback)

    def use(self, fn: Callable, filter_node: Optional[Callable] = None) -> 'Task':
        """
        Overview:
            Register middleware to task. The middleware will be executed by it's registry order.
        Arguments:
            - fn (:obj:`Callable`): A middleware is a function with only one argument: ctx.
        """
        if self.router.is_active and filter_node and not filter_node(self.router.node_id):
            return self
        self.middleware.append(fn)
        return self

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
    def forward(self, fn: Callable, ctx: Context = None, backward_stack: List[Generator] = None) -> 'Task':
        """
        Overview:
            This function will execute the middleware until the first yield statment,
            or the end of the middleware.
        Arguments:
            - fn (:obj:`Callable`): Function with contain the ctx argument in middleware.
        """
        if not backward_stack:
            backward_stack = self._backward_stack
        if not ctx:
            ctx = self.ctx
        g = fn(ctx)
        if isinstance(g, GeneratorType):
            try:
                next(g)
                backward_stack.append(g)
            except StopIteration:
                pass
        return self

    @enable_async
    def backward(self, backward_stack: List[Generator] = None) -> 'Task':
        """
        Overview:
            Execute the rest part of middleware, by the reversed order of registry.
        """
        if not backward_stack:
            backward_stack = self._backward_stack
        while backward_stack:
            # FILO
            g = backward_stack.pop()
            try:
                next(g)
            except StopIteration:
                continue
        return self

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

    def renew(self) -> 'Task':
        """
        Overview:
            Renew the context instance, this function should be called after backward in the end of iteration.
        """
        # Sync should be called before backward, otherwise it is possible
        # that some generators have not been pushed to backward_stack.
        self.sync()
        self.backward()
        self.sync()
        # Renew context
        old_ctx = self.ctx
        if self.router.is_active:
            # Send context to other parallel processes
            self.async_executor(self.router.send_rpc, "task.emit", "sync_parallel_ctx", old_ctx)

        new_ctx = old_ctx.renew()
        new_ctx.total_step = old_ctx.total_step + 1
        self.ctx = new_ctx
        return self

    def __enter__(self) -> "Task":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def stop(self) -> None:
        if self._thread_pool:
            self._thread_pool.shutdown()

    def sync(self) -> 'Task':
        if self._loop:
            self._loop.run_until_complete(self.sync_tasks())
        return self

    async def sync_tasks(self) -> Awaitable[None]:
        while self._async_stack:
            # FIFO
            t = self._async_stack.pop(0)
            await t

    def wait_for_attach_callback(self, attach_callback: Callable, n_timeout: int = 30):
        if len(self.router.attach_to) > 0:
            logging.warning(
                "The attach mode will wait for the latest context, an exception will \
be thrown after the timeout {}s is reached".format(n_timeout)
            )
            is_timeout = True
            ctx = None

            def on_sync_parallel_ctx(new_ctx):
                nonlocal ctx
                ctx = new_ctx

            self.once("sync_parallel_ctx", on_sync_parallel_ctx)
            for _ in range(n_timeout * 10):
                if ctx:
                    is_timeout = False
                    break
                time.sleep(0.1)
            if is_timeout:
                # If attach callback is defined, the attach mode should wait for callback finished,
                # otherwise it may overwrite the training results of other processes
                raise TimeoutError("Attach timeout, not received the latest context.")
            attach_callback(ctx)

    def async_executor(self, fn: Callable, *args, **kwargs) -> None:
        """
        Overview:
            Execute task in background, then apppend the future instance in _async_stack.
        Arguments:
            - fn (:obj:`Callable`): Synchronization fuction.
        """
        if not self._loop:
            raise Exception("Event loop was not initialized, please call this function in async or parallel mode")
        t = self._loop.run_in_executor(self._thread_pool, fn, *args, **kwargs)
        self._async_stack.append(t)

    def emit(self, event_name, *args, **kwargs):
        if event_name in self.event_listeners:
            for fn in self.event_listeners[event_name]:
                fn(*args, **kwargs)
        if event_name in self.once_listeners:
            while self.once_listeners[event_name]:
                fn = self.once_listeners[event_name].pop()
                fn(*args, **kwargs)

    def on(self, event: str, fn: Callable) -> None:
        self.event_listeners[event].append(fn)

    def once(self, event: str, fn: Callable) -> None:
        self.once_listeners[event].append(fn)

    @property
    def finish(self) -> bool:
        """
        Overview:
            Link the ctx's finish state, in order to be easily called externally.
        """
        return self.ctx._finish

    def __copy__(self):
        return Task(**self.__dict__)
