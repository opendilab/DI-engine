import logging
import copy
import time
import asyncio
import concurrent.futures
from types import GeneratorType
from typing import Awaitable, Callable, Generator, List, Optional
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
            n_async_workers: Optional[int] = None,
            parallel_mode: bool = False,
            router: Optional['Parallel'] = None,
            n_parallel_workers: Optional[int] = None,
            attach_to: List[str] = None,
            protocol: str = "ipc",
            middleware: List[Callable] = None,
            bind_address: str = None,
            bind_ports: Optional[List[int]] = None,
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
        if async_mode or parallel_mode:
            self._thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=n_async_workers)
            self._loop = asyncio.new_event_loop()

        # Parallel segment
        self.parallel_mode = parallel_mode
        if parallel_mode:
            self.router = router or Parallel()
            assert n_parallel_workers, "The number of workers must be specified in parallel mode!"
            self.n_parallel_workers = n_parallel_workers
            self.attach_to = attach_to or []
            self.protocol = protocol
            self.bind_address = bind_address
            self.bind_ports = bind_ports

    def use(self, fn: Callable) -> 'Task':
        """
        Overview:
            Register middleware to task. The middleware will be executed by it's registry order.
        Arguments:
            - fn (:obj:`Callable`): A middleware is a function with only one argument: ctx.
        """
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

        def _execute(task: Task):
            if len(task.middleware) == 0:
                return
            while task.ctx.total_step < max_step:
                for fn in task.middleware:
                    task.forward(fn)
                task.backward()
                if task.finish:
                    break
                task.renew()

        if self.parallel_mode:
            self.parallel(_execute)
        else:
            _execute(self)

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
        if self.parallel_mode and self.router:
            # Send context to other parallel processes
            # The maximum number of iterations is estimated from the total number of all processes
            # Set current total step to real step in the cluster
            prev_total_step = old_ctx.get("prev") and old_ctx.get("prev").total_step or -1
            old_ctx.total_step = max(old_ctx.total_step, prev_total_step + 1)
            self.async_executor(self.router.send_rpc, "sync_parallel_ctx", old_ctx)

        new_ctx = old_ctx.renew()
        new_ctx.total_step = old_ctx.total_step + 1
        new_ctx.prev = self._inherit_ctx(old_ctx, old_ctx.prev) if old_ctx.get("prev") else old_ctx
        self.ctx = new_ctx
        return self

    def step(self) -> 'Task':
        return self

    def __enter__(self) -> 'Task':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.renew()

    def sync(self) -> 'Task':
        if self._loop:
            self._loop.run_until_complete(self.sync_tasks())
        return self

    async def sync_tasks(self) -> Awaitable[None]:
        while self._async_stack:
            # FIFO
            t = self._async_stack.pop(0)
            await t

    def parallel(self, main_process: Callable):

        def _parallel():
            task = copy.copy(self)
            task.router.register_rpc("sync_parallel_ctx", task.sync_parallel_ctx)
            n_timeout = 30
            if len(task.attach_to) > 0:
                logging.warning(
                    "The attach mode will wait for the latest context, an exception will \
be thrown after the timeout {}s is reached".format(n_timeout)
                )
                is_timeout = True
                for _ in range(n_timeout * 10):
                    if task.ctx.get("prev"):
                        is_timeout = False
                        task.ctx = task.ctx.prev
                        break
                    time.sleep(0.1)
                if is_timeout:
                    # The attach mode does not allow to start from step 0 alone,
                    # otherwise it may overwrite the training results of other processes
                    raise TimeoutError("Attach timeout, not received the latest context.")
            main_process(task)

        self.router.run(
            _parallel,
            n_workers=self.n_parallel_workers,
            attach_to=self.attach_to,
            protocol=self.protocol,
            address=self.bind_address,
            ports=self.bind_ports
        )

    def sync_parallel_ctx(self, ctx: Context):
        if self.ctx.get("prev") and self.ctx.prev.total_step > ctx.total_step:
            self.ctx.prev = self._inherit_ctx(self.ctx.prev, ctx)
        else:
            self.ctx.prev = self._inherit_ctx(ctx, self.ctx.get("prev") or Context())

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

    @property
    def finish(self) -> bool:
        """
        Overview:
            Link the ctx's finish state, in order to be easily called externally.
        """
        return self.ctx._finish

    def _inherit_ctx(self, new_: Context, old: Context) -> Context:
        """
        Overview:
            Overwrite old context with new properies, If the key does not exist in the new context,
            the attributes in old are retained.
        Arguments:
            - new_ (:obj:`Context`): New context.
            - old (:obj:`Context`): Old context.
        Returns:
            - child (:obj:`Context`): The heir.
        """
        for key, value in new_.items():
            old[key] = value
        if "prev" in old:
            del old["prev"]
        return old

    def __copy__(self):
        return Task(**self.__dict__)
