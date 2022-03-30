from time import sleep
import pytest

import multiprocessing as mp
from ding.framework.message_queue.nng import NNGMQ


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
