from asyncio import InvalidStateError
from asyncio.tasks import FIRST_EXCEPTION
from collections import defaultdict
import time
import asyncio
import concurrent.futures
import fnmatch
import math
from types import GeneratorType
from typing import Any, Awaitable, Callable, Dict, Generator, Iterable, List, Optional, Set, Union
from ding.framework.context import Context
from ding.framework.parallel import Parallel
from ding.framework.event_loop import EventLoop
from functools import wraps


def enable_async(func: Callable) -> Callable:
    """
    Overview:
        Empower the function with async ability.
    Arguments:
        - func (:obj:`Callable`): The original function.
    Returns:
        - runtime_handler (:obj:`Callable`): The wrap function.
    """

    @wraps(func)
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
            t = task._async_loop.run_in_executor(task._thread_pool, func, task, *args, **kwargs)
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
            n_async_workers: int = 3,
            middleware: Optional[List[Callable]] = None,
            step_wrappers: Optional[List[Callable]] = None,
            labels: Optional[Set[str]] = None,
            **_
    ) -> None:
        self._finish = False
        self.middleware = middleware or []
        self.step_wrappers = step_wrappers or []
        self.ctx = Context()
        self.parallel_ctx = Context()
        self._backward_stack = []
        # Bind event loop functions
        self._event_loop = EventLoop.get_event_loop("task_{}".format(id(self)))
        self.on = self._event_loop.on
        self.once = self._event_loop.once
        self._emit = self._event_loop.emit
        self.off = self._event_loop.off

        # Async segment
        self.async_mode = async_mode
        self.n_async_workers = n_async_workers
        self._async_stack = []
        self._async_loop = None
        self._thread_pool = None
        self._exception = None
        self.labels = labels or set()

        # Parallel segment
        self.router = Parallel()
        if async_mode or self.router.is_active:
            self._thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=n_async_workers)
            self._async_loop = asyncio.new_event_loop()

        if self.router.is_active:
            self.router.register_rpc("task._emit", self._emit)

            def sync_finish(value):
                self._finish = value

            self.on("finish", sync_finish)

        self.init_labels()

    def init_labels(self):
        if self.async_mode:
            self.labels.add("async")
        if self.router.is_active:
            self.labels.add("distributed")
            self.labels.add("node.{}".format(self.router.node_id))
            for label in self.router.labels:
                self.labels.add(label)
        else:
            self.labels.add("standalone")

    def use(self, fn: Callable, filter_labels: Optional[Iterable[str]] = None) -> 'Task':
        """
        Overview:
            Register middleware to task. The middleware will be executed by it's registry order.
        Arguments:
            - fn (:obj:`Callable`): A middleware is a function with only one argument: ctx.
        """
        if not filter_labels or self.match_labels(filter_labels):
            self.middleware.append(fn)
        return self

    def use_step_wrapper(self, fn: Callable) -> 'Task':
        """
        Overview:
            Register wrappers to task. A wrapper works like a decorator, but task will apply this \
            decorator on top of each middleware.
        Arguments:
            - fn (:obj:`Callable`): A wrapper is a decorator, so the first argument is a callable function.
        """
        self.step_wrappers.append(fn)
        return self

    def match_labels(self, patterns: Union[Iterable[str], str]) -> bool:
        """
        Overview:
            A list of patterns to match labels.
        Arguments:
            - patterns (:obj:`Union[Iterable[str], str]`): Glob like pattern, e.g. node.1, node.*.
        """
        if isinstance(patterns, str):
            patterns = [patterns]
        return any([fnmatch.filter(self.labels, p) for p in patterns])

    def run(self, max_step: int = int(1e10)) -> None:
        """
        Overview:
            Execute the iterations, when reach the max_step or task.finish is true,
            The loop will be break.
        Arguments:
            - max_step (:obj:`int`): Max step of iterations.
        """
        if len(self.middleware) == 0:
            return
        for i in range(max_step):
            for fn in self.middleware:
                self.forward(fn)
            # Sync should be called before backward, otherwise it is possible
            # that some generators have not been pushed to backward_stack.
            self.sync()
            self.backward()
            self.sync()
            if i == max_step - 1:
                self.finish = True
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
        for wrapper in self.step_wrappers:
            fn = wrapper(fn)
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

        name = ",".join([fn.__name__ for fn in fns])
        _sequence.__name__ = "sequence<{}>".format(name)
        return _sequence

    def renew(self) -> 'Task':
        """
        Overview:
            Renew the context instance, this function should be called after backward in the end of iteration.
        """
        # Renew context
        old_ctx = self.ctx
        new_ctx = old_ctx.renew()
        new_ctx.total_step = old_ctx.total_step + 1
        self.ctx = new_ctx
        return self

    def __enter__(self) -> "Task":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def stop(self) -> None:
        """
        Overview:
            Stop and cleanup every thing in the runtime of task.
        """
        if self._thread_pool:
            self._thread_pool.shutdown()
        self._event_loop.stop()
        # The middleware and listeners may contain some methods that reference to task,
        # If we do not clear them after the task exits, we may find that gc will not clean up the task object.
        self.middleware.clear()
        self.step_wrappers.clear()
        self._backward_stack.clear()
        self._async_stack.clear()

    def sync(self) -> 'Task':
        if self._async_loop:
            self._async_loop.run_until_complete(self.sync_tasks())
        return self

    async def sync_tasks(self) -> Awaitable[None]:
        if self._async_stack:
            await asyncio.wait(self._async_stack, return_when=FIRST_EXCEPTION)
            while self._async_stack:
                t = self._async_stack.pop(0)
                try:
                    e = t.exception()
                    if e:
                        self._exception = e
                        raise e
                except InvalidStateError:
                    # Not finished. https://docs.python.org/3/library/asyncio-task.html#asyncio.Task.exception
                    pass

    def async_executor(self, fn: Callable, *args, **kwargs) -> None:
        """
        Overview:
            Execute task in background, then apppend the future instance in _async_stack.
        Arguments:
            - fn (:obj:`Callable`): Synchronization fuction.
        """
        if not self._async_loop:
            raise Exception("Event loop was not initialized, please call this function in async or parallel mode")
        t = self._async_loop.run_in_executor(self._thread_pool, fn, *args, **kwargs)
        self._async_stack.append(t)

    def emit(self, event: str, *args, **kwargs) -> None:
        """
        Overview:
            Emit an event, call listeners.
        Arguments:
            - event (:obj:`str`): Event name.
            - only_remote (:obj:`bool`): Only broadcast the event to the connected nodes, default is False.
            - only_local (:obj:`bool`): Only emit local event, default is False.
            - args (:obj:`any`): Rest arguments for listeners.
        """
        # Check if need to broadcast event to connected nodes, default is True
        if kwargs.get("only_local"):
            kwargs.pop("only_local")
            self._emit(event, *args, **kwargs)
        elif kwargs.get("only_remote"):
            kwargs.pop("only_remote")
            if self.router.is_active:
                self.async_executor(self.router.send_rpc, "task._emit", event, *args, **kwargs)
        else:
            if self.router.is_active:
                self.async_executor(self.router.send_rpc, "task._emit", event, *args, **kwargs)
            self._emit(event, *args, **kwargs)

    def wait_for(self, event: str, timeout: float = math.inf, ignore_timeout_exception: bool = True) -> Any:
        """
        Overview:
            Wait for an event and block the thread.
        Arguments:
            - event (:obj:`str`): Event name.
            - timeout (:obj:`float`): Timeout in seconds.
            - ignore_timeout_exception (:obj:`bool`): If this is False, an exception will occur when meeting timeout.
        """
        received = False
        result = None

        def _receive_event(*args, **kwargs):
            nonlocal result, received
            result = (args, kwargs)
            received = True

        self.once(event, _receive_event)

        start = time.time()
        while time.time() - start < timeout:
            if received or self._exception:
                return result
            time.sleep(0.01)

        if ignore_timeout_exception:
            return result
        else:
            raise TimeoutError("Timeout when waiting for event: {}".format(event))

    def __copy__(self):
        return Task(**self.__dict__)

    @property
    def finish(self):
        return self._finish

    @finish.setter
    def finish(self, value: bool):
        self._finish = value
        if self.router.is_active and value is True:
            self.emit("finish", value)
