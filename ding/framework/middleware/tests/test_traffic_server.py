import pytest
import os
import shutil
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
    dir = "./traffic_server_test_parallel/"
    with task.start(async_mode=True):
        if task.router.node_id > 0:
            traffic.set_config(is_writer=True, file_path=dir + "log.txt", router=Parallel())
            task.use(traffic_server())
        else:
            traffic.set_config(router=Parallel())
            task.use(fn_record_something())
        task.run(max_step=10)


def main():
    dir = "./traffic_server_test_local/"
    with task.start(async_mode=True):
        traffic.set_config(is_writer=True, file_path=dir + "log.txt")
        task.use(traffic_server())
        task.use(fn_record_something())
        task.run(max_step=10)
    traffic.close()


@pytest.mark.unittest
class TestTrafficServerModule:

    def test_traffic_server_parallel_mode(self):
        Parallel.runner(n_parallel_workers=2, topology="mesh")(parallel_main)
        clean_up("./traffic_server_test_parallel/")

    def test_traffic_server_local_mode(self):
        main()
        clean_up("./traffic_server_test_local/")
