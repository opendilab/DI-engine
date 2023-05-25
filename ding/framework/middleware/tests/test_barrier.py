import random
import time
import socket
import pytest
import multiprocessing as mp
from ditk import logging
from ding.framework import task
from ding.framework.parallel import Parallel
from ding.framework.context import OnlineRLContext
from ding.framework.middleware.barrier import Barrier

PORTS_LIST = ["1235", "1236", "1237"]


class EnvStepMiddleware:

    def __call__(self, ctx):
        yield
        ctx.env_step += 1


class SleepMiddleware:

    def __init__(self, node_id):
        self.node_id = node_id

    def random_sleep(self, diection, step):
        random.seed(self.node_id + step)
        sleep_second = random.randint(1, 5)
        logging.info("Node:[{}] env_step:[{}]-{} will sleep:{}s".format(self.node_id, step, diection, sleep_second))
        for i in range(sleep_second):
            time.sleep(1)
            print("Node:[{}] sleepping...".format(self.node_id))
        logging.info("Node:[{}] env_step:[{}]-{} wake up!".format(self.node_id, step, diection))

    def __call__(self, ctx):
        self.random_sleep("forward", ctx.env_step)
        yield
        self.random_sleep("backward", ctx.env_step)


def star_barrier():
    with task.start(ctx=OnlineRLContext()):
        node_id = task.router.node_id
        if node_id == 0:
            attch_from_nums = 3
        else:
            attch_from_nums = 0
        barrier = Barrier(attch_from_nums)
        task.use(barrier, lock=False)
        task.use(SleepMiddleware(node_id), lock=False)
        task.use(barrier, lock=False)
        task.use(EnvStepMiddleware(), lock=False)
        try:
            task.run(2)
        except Exception as e:
            logging.error(e)
            assert False


def mesh_barrier():
    with task.start(ctx=OnlineRLContext()):
        node_id = task.router.node_id
        attch_from_nums = 3 - task.router.node_id
        barrier = Barrier(attch_from_nums)
        task.use(barrier, lock=False)
        task.use(SleepMiddleware(node_id), lock=False)
        task.use(barrier, lock=False)
        task.use(EnvStepMiddleware(), lock=False)
        try:
            task.run(2)
        except Exception as e:
            logging.error(e)
            assert False


def unmatch_barrier():
    with task.start(ctx=OnlineRLContext()):
        node_id = task.router.node_id
        attch_from_nums = 3 - task.router.node_id
        task.use(Barrier(attch_from_nums, 5), lock=False)
        if node_id != 2:
            task.use(Barrier(attch_from_nums, 5), lock=False)
        try:
            task.run(2)
        except TimeoutError as e:
            assert node_id != 2
            logging.info("Node:[{}] timeout with barrier".format(node_id))
        else:
            time.sleep(5)
            assert node_id == 2
            logging.info("Node:[{}] finish barrier".format(node_id))


def launch_barrier(args):
    i, topo, fn, test_id = args
    address = socket.gethostbyname(socket.gethostname())
    topology = "alone"
    attach_to = []
    port_base = PORTS_LIST[test_id]
    port = port_base + str(i)
    if topo == 'star':
        if i != 0:
            attach_to = ['tcp://{}:{}{}'.format(address, port_base, 0)]
    elif topo == 'mesh':
        for j in range(i):
            attach_to.append('tcp://{}:{}{}'.format(address, port_base, j))

    Parallel.runner(
        node_ids=i,
        ports=int(port),
        attach_to=attach_to,
        topology=topology,
        protocol="tcp",
        n_parallel_workers=1,
        startup_interval=0
    )(fn)


@pytest.mark.unittest
def test_star_topology_barrier():
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=4) as pool:
        pool.map(launch_barrier, [[i, 'star', star_barrier, 0] for i in range(4)])
        pool.close()
        pool.join()


@pytest.mark.unittest
def test_mesh_topology_barrier():
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=4) as pool:
        pool.map(launch_barrier, [[i, 'mesh', mesh_barrier, 1] for i in range(4)])
        pool.close()
        pool.join()


@pytest.mark.unittest
def test_unmatch_barrier():
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=4) as pool:
        pool.map(launch_barrier, [[i, 'mesh', unmatch_barrier, 2] for i in range(4)])
        pool.close()
        pool.join()
