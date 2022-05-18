import pytest
from ding.utils import traffic
from ding.framework import task, Context
from ding.framework import Parallel
from ding.framework.middleware import traffic_server


def clean_up(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)


def fn_record_something():

    def _fn(ctx: "Context"):
        traffic.record(data_str="str", data_int=1, data_float=0.5)

    return _fn


def parallel_main():
    with task.start(async_mode=True):
        if task.router.node_id > 0:
            traffic.set_config(is_writer=True, file_path="./traffic_test/log.txt", router=Parallel())
            task.use(traffic_server())
        else:
            traffic.set_config(router=Parallel())
            task.use(fn_record_something())
        task.run(max_step=10)
    clean_up("./traffic_test/log.txt")


def main():
    with task.start(async_mode=True):
        traffic.set_config(is_writer=True, file_path="./traffic_test/log.txt")
        task.use(traffic_server())
        task.use(fn_record_something())
        task.run(max_step=10)
    clean_up("./traffic_test/log.txt")


@pytest.mark.unittest
class TestTrafficServerModule:

    def test_traffic_server_parallel_mode(self):
        Parallel.runner(n_parallel_workers=2, topology="mesh")(parallel_main)

    def test_traffic_server_local_mode(self):
        main()
