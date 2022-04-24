from asyncio import InvalidStateError
from asyncio.tasks import FIRST_EXCEPTION
from collections import OrderedDict
from threading import Lock
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
    def runtime_handler(task: "Task", *args, async_mode: Optional[bool] = None, **kwargs) -> "Task":
        """
        Overview:
            If task's async mode is enabled, execute the step in current loop executor asyncly,
            or execute the task sync.
        Arguments:
            - task (:obj:`Task`): The task instance.
            - async_mode (:obj:`Optional[bool]`): Whether using async mode.
        Returns:
            - result (:obj:`Union[Any, Awaitable]`): The result or future object of middleware.
        """
        if async_mode is None:
            async_mode = task.async_mode
        if async_mode:
            assert not kwargs, "Should not use kwargs in async_mode, use position parameters, kwargs: {}".format(kwargs)
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

    def start(
            self,
            async_mode: bool = False,
            n_async_workers: int = 3,
            ctx: Optional[Context] = None,
            labels: Optional[Set[str]] = None
    ) -> "Task":
        # This flag can be modified by external or associated processes
        self._finish = False
        # This flag can only be modified inside the class, it will be set to False in the end of stop
        self._running = True
        self._middleware = []
        self._wrappers = []
        self.ctx = ctx or Context()
        self._backward_stack = OrderedDict()
        # Bind event loop functions
        self._event_loop = EventLoop("task_{}".format(id(self)))

        # Async segment
        self.async_mode = async_mode
        self.n_async_workers = n_async_workers
        self._async_stack = []
        self._async_loop = None
        self._thread_pool = None
        self._exception = None
        self._thread_lock = Lock()
        self.labels = labels or set()

        # Parallel segment
        self.router = Parallel()
        if async_mode or self.router.is_active:
            self._activate_async()

        if self.router.is_active:

            def sync_finish(value):
                self._finish = value

            self.on("finish", sync_finish)

        self.init_labels()
        return self

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

    def use(self, fn: Callable, lock: Union[bool, Lock] = False) -> 'Task':
        """
        Overview:
            Register middleware to task. The middleware will be executed by it's registry order.
        Arguments:
            - fn (:obj:`Callable`): A middleware is a function with only one argument: ctx.
            - lock (:obj:`Union[bool, Lock]`): There can only be one middleware execution under lock at any one time.
        Returns:
            - task (:obj:`Task`): The task.
        """
        for wrapper in self._wrappers:
            fn = wrapper(fn)
        self._middleware.append(self.wrap(fn, lock=lock))
        return self

    def use_wrapper(self, fn: Callable) -> 'Task':
        """
        Overview:
            Register wrappers to task. A wrapper works like a decorator, but task will apply this \
            decorator on top of each middleware.
        Arguments:
            - fn (:obj:`Callable`): A wrapper is a decorator, so the first argument is a callable function.
        Returns:
            - task (:obj:`Task`): The task.
        """
        # Wrap exist middlewares
        for i, middleware in enumerate(self._middleware):
            self._middleware[i] = fn(middleware)
        self._wrappers.append(fn)
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

    def run(self, max_step: int = int(1e12)) -> None:
        """
        Overview:
            Execute the iterations, when reach the max_step or task.finish is true,
            The loop will be break.
        Arguments:
            - max_step (:obj:`int`): Max step of iterations.
        """
        assert self._running, "Please make sure the task is running before calling the this method, see the task.start"
        if len(self._middleware) == 0:
            return
        for i in range(max_step):
            for fn in self._middleware:
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

    def wrap(self, fn: Callable, lock: Union[bool, Lock] = False) -> Callable:
        """
        Overview:
            Wrap the middleware, make it can be called directly in other middleware.
        Arguments:
            - fn (:obj:`Callable`): The middleware.
            - lock (:obj:`Union[bool, Lock]`): There can only be one middleware execution under lock at any one time.
        Returns:
            - fn_back (:obj:`Callable`): It will return a backward function, which will call the rest part of
                the middleware after yield. If this backward function was not called, the rest part of the middleware
                will be called in the global backward step.
        """
        if lock is True:
            lock = self._thread_lock

        @wraps(fn)
        def forward(ctx: Context):
            if lock:
                with lock:
                    g = self.forward(fn, ctx, async_mode=False)
            else:
                g = self.forward(fn, ctx, async_mode=False)

            def backward():
                backward_stack = OrderedDict()
                key = id(g)
                backward_stack[key] = self._backward_stack.pop(key)
                if lock:
                    with lock:
                        self.backward(backward_stack, async_mode=False)
                else:
                    self.backward(backward_stack, async_mode=False)

            return backward

        return forward

    @enable_async
    def forward(self, fn: Callable, ctx: Optional[Context] = None) -> Optional[Generator]:
        """
        Overview:
            This function will execute the middleware until the first yield statment,
            or the end of the middleware.
        Arguments:
            - fn (:obj:`Callable`): Function with contain the ctx argument in middleware.
            - ctx (:obj:`Optional[Context]`): Replace global ctx with a customized ctx.
        Returns:
            - g (:obj:`Optional[Generator]`): The generator if the return value of fn is a generator.
        """
        assert self._running, "Please make sure the task is running before calling the this method, see the task.start"
        if not ctx:
            ctx = self.ctx
        g = fn(ctx)
        if isinstance(g, GeneratorType):
            try:
                next(g)
                self._backward_stack[id(g)] = g
                return g
            except StopIteration:
                pass

    @enable_async
    def backward(self, backward_stack: Optional[Dict[str, Generator]] = None) -> None:
        """
        Overview:
            Execute the rest part of middleware, by the reversed order of registry.
        Arguments:
            - backward_stack (:obj:`Optional[Dict[str, Generator]]`): Replace global backward_stack with a customized \
                stack.
        """
        assert self._running, "Please make sure the task is running before calling the this method, see the task.start"
        if not backward_stack:
            backward_stack = self._backward_stack
        while backward_stack:
            # FILO
            _, g = backward_stack.popitem()
            try:
                next(g)
            except StopIteration:
                continue

    def serial(self, *fns: List[Callable]) -> Callable:
        """
        Overview:
            Wrap functions and keep them run in serial, Usually in order to avoid the confusion
            of dependencies in async mode.
        Arguments:
            - fn (:obj:`Callable`): Chain a serial of middleware, wrap them into one middleware function.
        """

        def _serial(ctx):
            backward_keys = []
            for fn in fns:
                g = self.forward(fn, ctx, async_mode=False)
                if isinstance(g, GeneratorType):
                    backward_keys.append(id(g))
            yield
            backward_stack = OrderedDict()
            for k in backward_keys:
                backward_stack[k] = self._backward_stack.pop(k)
            self.backward(backward_stack=backward_stack, async_mode=False)

        name = ",".join([fn.__name__ for fn in fns])
        _serial.__name__ = "serial<{}>".format(name)
        return _serial

    def parallel(self, *fns: List[Callable]) -> Callable:
        """
        Overview:
            Wrap functions and keep them run in parallel, should not use this funciton in async mode.
        Arguments:
            - fn (:obj:`Callable`): Parallelized middleware, wrap them into one middleware function.
        """
        self._activate_async()

        def _parallel(ctx):
            backward_keys = []
            for fn in fns:
                g = self.forward(fn, ctx, async_mode=True)
                if isinstance(g, GeneratorType):
                    backward_keys.append(id(g))
            self.sync()
            yield
            backward_stack = OrderedDict()
            for k in backward_keys:
                backward_stack[k] = self._backward_stack.pop(k)
            self.backward(backward_stack, async_mode=True)
            self.sync()

        name = ",".join([fn.__name__ for fn in fns])
        _parallel.__name__ = "parallel<{}>".format(name)
        return _parallel

    def renew(self) -> 'Task':
        """
        Overview:
            Renew the context instance, this function should be called after backward in the end of iteration.
        """
        assert self._running, "Please make sure the task is running before calling the this method, see the task.start"
        self.ctx = self.ctx.renew()
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
        self.router.off(self._wrap_event_name("*"))
        if self._async_loop:
            self._async_loop.stop()
            self._async_loop.close()
        # The middleware and listeners may contain some methods that reference to task,
        # If we do not clear them after the task exits, we may find that gc will not clean up the task object.
        self._middleware.clear()
        self._wrappers.clear()
        self._backward_stack.clear()
        self._async_stack.clear()
        self._running = False

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

    def emit(self, event: str, *args, only_remote: bool = False, only_local: bool = False, **kwargs) -> None:
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
        assert self._running, "Please make sure the task is running before calling the this method, see the task.start"
        if only_local:
            self._event_loop.emit(event, *args, **kwargs)
        elif only_remote:
            if self.router.is_active:
                self.async_executor(self.router.emit, self._wrap_event_name(event), event, *args, **kwargs)
        else:
            if self.router.is_active:
                self.async_executor(self.router.emit, self._wrap_event_name(event), event, *args, **kwargs)
            self._event_loop.emit(event, *args, **kwargs)

    def on(self, event: str, fn: Callable) -> None:
        """
        Overview:
            Subscribe to an event, execute this function every time the event is emitted.
        Arguments:
            - event (:obj:`str`): Event name.
            - fn (:obj:`Callable`): The function.
        """
        self._event_loop.on(event, fn)
        if self.router.is_active:
            self.router.on(self._wrap_event_name(event), self._event_loop.emit)

    def once(self, event: str, fn: Callable) -> None:
        """
        Overview:
            Subscribe to an event, execute this function only once when the event is emitted.
        Arguments:
            - event (:obj:`str`): Event name.
            - fn (:obj:`Callable`): The function.
        """
        self._event_loop.once(event, fn)
        if self.router.is_active:
            self.router.on(self._wrap_event_name(event), self._event_loop.emit)

    def off(self, event: str, fn: Optional[Callable] = None) -> None:
        """
        Overview:
            Unsubscribe an event
        Arguments:
            - event (:obj:`str`): Event name.
            - fn (:obj:`Callable`): The function.
        """
        self._event_loop.off(event, fn)
        if self.router.is_active:
            self.router.off(self._wrap_event_name(event))

    def wait_for(self, event: str, timeout: float = math.inf, ignore_timeout_exception: bool = True) -> Any:
        """
        Overview:
            Wait for an event and block the thread.
        Arguments:
            - event (:obj:`str`): Event name.
            - timeout (:obj:`float`): Timeout in seconds.
            - ignore_timeout_exception (:obj:`bool`): If this is False, an exception will occur when meeting timeout.
        """
        assert self._running, "Please make sure the task is running before calling the this method, see the task.start"
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

    @property
    def finish(self):
        return self._finish

    @finish.setter
    def finish(self, value: bool):
        self._finish = value
        if self.router.is_active and value is True:
            self.emit("finish", value)

    def _wrap_event_name(self, event: str) -> str:
        """
        Overview:
            Wrap the event name sent to the router.
        Arguments:
            - event (:obj:`str`): Event name
        """
        return "task.{}".format(event)

    def _activate_async(self):
        if not self._thread_pool:
            self._thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.n_async_workers)
        if not self._async_loop:
            self._async_loop = asyncio.new_event_loop()


task = Task()
