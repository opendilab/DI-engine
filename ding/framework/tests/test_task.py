import pytest
from threading import Lock
from time import sleep
import random
from ding.framework import task, Context, Parallel


@pytest.mark.unittest
def test_serial_pipeline():

    def step0(ctx):
        ctx.setdefault("pipeline", [])
        ctx.pipeline.append(0)

    def step1(ctx):
        ctx.pipeline.append(1)

    # Execute step1, step2 twice
    with task.start():
        for _ in range(2):
            task.forward(step0)
            task.forward(step1)
        assert task.ctx.pipeline == [0, 1, 0, 1]

        # Renew and execute step1, step2
        task.renew()
        assert task.ctx.total_step == 1
        task.forward(step0)
        task.forward(step1)
        assert task.ctx.pipeline == [0, 1]

        # Test context inheritance
        task.renew()


@pytest.mark.unittest
def test_serial_yield_pipeline():

    def step0(ctx):
        ctx.setdefault("pipeline", [])
        ctx.pipeline.append(0)
        yield
        ctx.pipeline.append(0)

    def step1(ctx):
        ctx.pipeline.append(1)

    with task.start():
        task.forward(step0)
        task.forward(step1)
        task.backward()
        assert task.ctx.pipeline == [0, 1, 0]
        assert len(task._backward_stack) == 0


@pytest.mark.unittest
def test_async_pipeline():

    def step0(ctx):
        ctx.setdefault("pipeline", [])
        ctx.pipeline.append(0)

    def step1(ctx):
        ctx.pipeline.append(1)

    # Execute step1, step2 twice
    with task.start(async_mode=True):
        for _ in range(2):
            task.forward(step0)
            sleep(0.1)
            task.forward(step1)
            sleep(0.1)
        task.backward()
        assert task.ctx.pipeline == [0, 1, 0, 1]
        task.renew()
        assert task.ctx.total_step == 1


@pytest.mark.unittest
def test_async_yield_pipeline():

    def step0(ctx):
        ctx.setdefault("pipeline", [])
        sleep(0.1)
        ctx.pipeline.append(0)
        yield
        ctx.pipeline.append(0)

    def step1(ctx):
        sleep(0.2)
        ctx.pipeline.append(1)

    with task.start(async_mode=True):
        task.forward(step0)
        task.forward(step1)
        sleep(0.3)
        task.backward().sync()
        assert task.ctx.pipeline == [0, 1, 0]
        assert len(task._backward_stack) == 0


def parallel_main():
    sync_count = 0

    def on_count():
        nonlocal sync_count
        sync_count += 1

    def counter(task):

        def _counter(ctx):
            sleep(0.2 + random.random() / 10)
            task.emit("count", only_remote=True)

        return _counter

    with task.start():
        task.on("count", on_count)
        task.use(counter(task))
        task.run(max_step=10)
        assert sync_count > 0


@pytest.mark.unittest
def test_parallel_pipeline():
    Parallel.runner(n_parallel_workers=2)(parallel_main)


@pytest.mark.unittest
def test_emit():
    with task.start():
        greets = []
        task.on("Greeting", lambda msg: greets.append(msg))

        def step1(ctx):
            task.emit("Greeting", "Hi")

        task.use(step1)
        task.run(max_step=10)
        sleep(0.1)
    assert len(greets) == 10


def emit_remote_main():
    with task.start():
        greets = []
        if task.router.node_id == 0:
            task.on("Greeting", lambda msg: greets.append(msg))
            for _ in range(20):
                if greets:
                    break
                sleep(0.1)
            assert len(greets) > 0
        else:
            for _ in range(20):
                task.emit("Greeting", "Hi", only_remote=True)
                sleep(0.1)
            assert len(greets) == 0


@pytest.mark.unittest
def test_emit_remote():
    Parallel.runner(n_parallel_workers=2)(emit_remote_main)


@pytest.mark.unittest
def test_wait_for():
    # Wait for will only work in async or parallel mode
    with task.start(async_mode=True, n_async_workers=2):
        greets = []

        def step1(_):
            hi = task.wait_for("Greeting")[0][0]
            if hi:
                greets.append(hi)

        def step2(_):
            task.emit("Greeting", "Hi")

        task.use(step1)
        task.use(step2)
        task.run(max_step=10)

    assert len(greets) == 10
    assert all(map(lambda hi: hi == "Hi", greets))

    # Test timeout exception
    with task.start(async_mode=True, n_async_workers=2):

        def step1(_):
            task.wait_for("Greeting", timeout=0.3, ignore_timeout_exception=False)

        task.use(step1)
        with pytest.raises(TimeoutError):
            task.run(max_step=1)


@pytest.mark.unittest
def test_async_exception():
    with task.start(async_mode=True, n_async_workers=2):

        def step1(_):
            task.wait_for("any_event")  # Never end

        def step2(_):
            sleep(0.3)
            raise Exception("Oh")

        task.use(step1)
        task.use(step2)
        with pytest.raises(Exception):
            task.run(max_step=2)

        assert task.ctx.total_step == 0


def early_stop_main():
    with task.start():
        task.use(lambda _: sleep(0.5))
        if task.match_labels("node.0"):
            task.run(max_step=10)
        else:
            task.run(max_step=2)
        assert task.ctx.total_step < 7


@pytest.mark.unittest
def test_early_stop():
    Parallel.runner(n_parallel_workers=2)(early_stop_main)


@pytest.mark.unittest
def test_parallel_in_sequencial():
    result = []

    def fast(_):
        result.append("fast")

    def slow(_):
        sleep(0.1)
        result.append("slow")

    with task.start():
        task.use(lambda _: result.append("begin"))
        task.use(task.parallel(slow, fast))
        task.run(max_step=1)
        assert result == ["begin", "fast", "slow"]


@pytest.mark.unittest
def test_serial_in_parallel():
    result = []

    def fast(_):
        result.append("fast")

    def slow(_):
        sleep(0.1)
        result.append("slow")

    with task.start(async_mode=True):
        task.use(lambda _: result.append("begin"))
        task.use(task.serial(slow, fast))
        task.run(max_step=1)

        assert result == ["begin", "slow", "fast"]


@pytest.mark.unittest
def test_nested_middleware():
    """
    When there is a yield in the middleware,
    calling this middleware in another will lead to an unexpected result.
    Use task.forward or task.wrap can fix this problem.
    """
    result = []

    def child():

        def _child(ctx: Context):
            result.append(3 * ctx.total_step)
            yield
            result.append(2 + 3 * ctx.total_step)

        return _child

    def mother():
        _child = task.wrap(child())

        def _mother(ctx: Context):
            child_back = _child(ctx)
            result.append(1 + 3 * ctx.total_step)
            child_back()

        return _mother

    with task.start():
        task.use(mother())
        task.run(2)
        assert result == [0, 1, 2, 3, 4, 5]


@pytest.mark.unittest
def test_use_lock():

    def slow(ctx):
        sleep(0.1)
        ctx.result = "slow"

    def fast(ctx):
        ctx.result = "fast"

    with task.start(async_mode=True):
        # The lock will turn async middleware into serial
        task.use(slow, lock=True)
        task.use(fast, lock=True)
        task.run(1)
        assert task.ctx.result == "fast"

    # With custom lock, it will not affect the inner lock of task
    lock = Lock()

    def slowest(ctx):
        sleep(0.3)
        ctx.result = "slowest"

    with task.start(async_mode=True):
        task.use(slow, lock=lock)
        # If it receives other locks, it will not be the last one to finish execution
        task.use(slowest, lock=True)
        task.use(fast, lock=lock)
        task.run(1)
        assert task.ctx.result == "slowest"
