import sys
import os
import time
from ditk import logging
import argparse
import tempfile

from random import random
from string import ascii_lowercase
from ding.framework import Parallel

alphabet = [c.encode('ascii') for c in ascii_lowercase]


class EasyCounter:

    def __init__(self):
        self._last = None
        self._cnt = 0

    def add(self, item):
        self._last = item
        self._cnt += 1

    def cnt(self):
        return self._cnt

    def last(self):
        return self._last


class SockTest:

    # In this class, we define three processes except the main process,
    # which are receiver, testee, and sender.
    # The testee receive messages from the sender, and sends its own greeting
    # messages to the receiver periodically.
    # During the test, we breakdown the network of testee, and then find out
    # what happens to the testee.

    @classmethod
    def receiver(cls, epoch, interval):
        router = Parallel()
        greets = EasyCounter()
        router.on("greeting_receiver", lambda msg: greets.add(msg))
        start_t = time.time()
        logging.info("receiver start ...")

        for i in range(epoch):
            while time.time() - start_t < i * interval:
                time.sleep(0.01)

            if greets.cnt() == 0 or i % 10 != 0:
                continue
            last_msg = greets.last()
            msg_idx, msg_t = last_msg.split("_")[-2:]
            logging.info(
                "receiver passed {:.2f} s, received {} msgs. last msg: idx {}, time {} s".format(
                    time.time() - start_t, greets.cnt(), msg_idx, msg_t
                )
            )

        logging.info("receiver done! total msg: {}".format(greets.cnt()))

    @classmethod
    def testee(cls, epoch, interval, data_size):
        words = b''.join([alphabet[int(random() * 26)] for _ in range(1024 * 1024)]) * data_size
        print("msg length: {:.4f} MB".format(sys.getsizeof(words) / 1024 / 1024))

        router = Parallel()
        greets = EasyCounter()
        router.on("greeting_testee", lambda msg: greets.add(msg))
        start_t = time.time()
        logging.info("testee start ...")

        with tempfile.NamedTemporaryFile(prefix="pytmp_", dir="./") as itf:
            print("testee: write ip address to the tempfile:", itf.name)
            with open(itf.name, 'w') as ifd:
                ifd.write("{}\n".format(router.get_ip()))

            for i in range(epoch):
                while time.time() - start_t < i * interval:
                    time.sleep(0.01)

                if router._retries == 0:
                    router.emit("greeting_receiver", "{}_{}_{:.2f}".format(words, i, time.time() - start_t))
                elif router._retries == 1:
                    router.emit("greeting_receiver", "recovered_{}_{:.2f}".format(i, time.time() - start_t))
                else:
                    raise Exception("Failed too many times")

                if greets.cnt() == 0 or i % 10 != 0:
                    continue
                last_msg = greets.last()
                msg_idx, msg_t = last_msg.split("_")[-2:]
                logging.info(
                    "testee passed {:.2f} s, received {} msgs. last msg: idx {}, time {} s".format(
                        time.time() - start_t, greets.cnt(), msg_idx, msg_t
                    )
                )

        logging.info("testee done! total msg: {} retries: {}".format(greets.cnt(), router._retries))

    @classmethod
    def sender(cls, epoch, interval, data_size):
        words = b''.join([alphabet[int(random() * 26)] for _ in range(1024 * 1024)]) * data_size
        print("msg length: {:.4f} MB".format(sys.getsizeof(words) / 1024 / 1024))

        router = Parallel()
        start_t = time.time()
        logging.info("sender start ...")

        for i in range(epoch):
            while time.time() - start_t < i * interval:
                time.sleep(0.01)

            router.emit("greeting_testee", "{}_{}_{:.2f}".format(words, i, time.time() - start_t))

        logging.info("sender done!")

    @classmethod
    def main(cls, epoch=1000, interval=1.0, data_size=1, file="tmp_p1"):
        router = Parallel()
        if router.node_id == 0:
            cls.receiver(epoch, interval)
        elif router.node_id == 1:
            cls.testee(epoch, interval, data_size)
        elif router.node_id == 2:
            cls.sender(epoch, interval, data_size)
        else:
            raise Exception("Invalid node id")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-t', type=int, default=1200)
    parser.add_argument('--interval', '-i', type=float, default=0.1)
    parser.add_argument('--data_size', '-s', type=int, default=1)
    args = parser.parse_args()
    Parallel.runner(
        n_parallel_workers=3, protocol="tcp", topology="mesh", auto_recover=True, max_retries=1
    )(SockTest.main, args.epoch, args.interval, args.data_size)
