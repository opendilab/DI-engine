from time import sleep
import pytest

import multiprocessing as mp
from ding.framework.message_queue.nng import NNGMQ
from threading import Thread

NUM_PROCESS = 5
NUM_MEGS = 10


def nng_main(i):
    if i == 0:
        listen_to = "tcp://127.0.0.1:50515"
        attach_to = None
        mq = NNGMQ(listen_to=listen_to, attach_to=attach_to)
        mq.listen()
        for _ in range(10):
            mq.publish("t", b"data")
            sleep(0.1)
    else:
        listen_to = "tcp://127.0.0.1:50516"
        attach_to = ["tcp://127.0.0.1:50515"]
        mq = NNGMQ(listen_to=listen_to, attach_to=attach_to)
        mq.listen()
        topic, msg = mq.recv()
        assert topic == "t"
        assert msg == b"data"


@pytest.mark.unittest
@pytest.mark.execution_timeout(10)
def test_nng():
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=2) as pool:
        pool.map(nng_main, range(2))


# If this test fails, please mail wangguoteng@sensetime.com
def nng_rendezvous(i):
    from ditk import logging
    logging.getLogger().setLevel(logging.DEBUG)

    def recv_warrper(mq):
        for j in range(NUM_MEGS + 1):
            msg = mq.recv()
            if not msg:
                return
            topic, msg = msg
            assert topic == "t"
            assert msg == b"data"

    if i == 0:
        listen_to = "tcp://127.0.0.1:50510"
        attach_to = None
        mq = NNGMQ(listen_to=listen_to, attach_to=attach_to, world_size=NUM_PROCESS, node_id=i, debug=True)
        mq.listen()
        listener = Thread(target=mq.recv, name="mq_listener", daemon=True)
        listener.start()
        mq.barrier()
        logging.info("Node {} barrier 1 is ok".format(i))
        for _ in range(10):
            mq.publish("t", b"data")
            sleep(0.1)
        mq.barrier()
        logging.info("Node {} barrier 2 is ok".format(i))
        mq.stop()
        listener.join(timeout=1)
    else:
        listen_to = "tcp://127.0.0.1:5051{}".format(i)
        attach_to = ["tcp://127.0.0.1:50510"]

        # Create artificial delays
        import random
        random = random.Random(i)
        sleep(int(random.uniform(0, 20)))

        mq = NNGMQ(listen_to=listen_to, attach_to=attach_to, world_size=NUM_PROCESS, node_id=i, debug=True)
        mq.listen()
        listener = Thread(target=recv_warrper, name="mq_listener", daemon=True, args=(mq, ))
        # listener = Thread(target=mq.recv, name="mq_listener", daemon=True)
        listener.start()
        mq.barrier()
        logging.info("Node {} barrier 1 is ok".format(i))
        mq.barrier()
        logging.info("Node {} barrier 2 is ok".format(i))
        mq.stop()
        listener.join(timeout=1)


@pytest.mark.unittest
@pytest.mark.execution_timeout(20)
def test_nng_rendezvous():
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=NUM_PROCESS) as pool:
        pool.map(nng_rendezvous, range(NUM_PROCESS))
