import time
import logging
import argparse
import sys
import os
from ding.framework import Parallel

class EasyCounter:
    def __init__(self):
        self._last = None
        self._cnt = 0
    
    def add(self, a):
        self._last = a
        self._cnt += 1
    
    def cnt(self):
        return self._cnt
    
    def last(self):
        return self._last


class SockTest:

    @classmethod
    def main_p0(cls, epoch, interval):
        router = Parallel()
        greets = EasyCounter()
        router.on("greeting_0", lambda msg: greets.add(msg))
        start_t = time.time()
        logging.info("main_p0 start ...")

        for i in range(epoch):
            while time.time() - start_t < i * interval:
                time.sleep(0.01)

            if greets.cnt() == 0 or i % 10 != 0:
                continue
            last_msg = greets.last()
            msg_idx, msg_t = last_msg.split("_")[-2:]
            logging.info("main_p0 passed {:.2f} s, received {} msgs. last msg: idx {}, time {} s"\
                .format(time.time() - start_t, greets.cnt(), msg_idx, msg_t))

        logging.info("main_p0 done! total msg: {}".format(greets.cnt()))

    @classmethod
    def main_p1(cls, epoch, interval, data_size, tmp_file):
        s = "a" * 1024 * 1024 * data_size
        print("msg length: {:.4f} MB".format(sys.getsizeof(s) / 1024 / 1024))

        router = Parallel()
        greets = EasyCounter()
        router.on("greeting_1", lambda msg: greets.add(msg))
        start_t = time.time()
        logging.info("main_p1 start ...")

        with open(tmp_file, "w") as f:
            f.write("{}\n".format(router.get_ip()))

        for i in range(epoch):
            while time.time() - start_t < i * interval:
                time.sleep(0.01)

            if router._retries == 0:
                router.emit("greeting_0", "{}_{}_{:.2f}".format(s, i, time.time() - start_t))
            elif router._retries == 1:
                router.emit("greeting_0", "recovered_{}_{:.2f}".format(i, time.time() - start_t))
            else:
                raise Exception("Failed too many times")

            if greets.cnt() == 0 or i % 10 != 0:
                continue
            last_msg = greets.last()
            msg_idx, msg_t = last_msg.split("_")[-2:]
            logging.info("main_p1 passed {:.2f} s, received {} msgs. last msg: idx {}, time {} s"\
                .format(time.time() - start_t, greets.cnt(), msg_idx, msg_t))

        if os.path.exists(tmp_file):
            os.remove(tmp_file)

        logging.info("main_p1 done! total msg: {} retries: {}".format(greets.cnt(), router._retries))

    @classmethod
    def main_p2(cls, epoch, interval, data_size):
        s = "b" * 1024 * 1024 * data_size
        print("msg length: {:.4f} MB".format(sys.getsizeof(s) / 1024 / 1024))

        router = Parallel()
        start_t = time.time()
        logging.info("main_p2 start ...")

        for i in range(epoch):
            while time.time() - start_t < i * interval:
                time.sleep(0.01)

            router.emit("greeting_1", "{}_{}_{:.2f}".format(s, i, time.time() - start_t))

        logging.info("main_p2 done!")

    @classmethod
    def main(cls, epoch=1000, interval=1.0, data_size=1, file="tmp_p1"):
        router = Parallel()
        if router.node_id == 0:
            cls.main_p0(epoch, interval)
        elif router.node_id == 1:
            cls.main_p1(epoch, interval, data_size, file)
        elif router.node_id == 2:
            cls.main_p2(epoch, interval, data_size)
        else:
            raise Exception("Invalid node id")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-t', type=int, default=1200)
    parser.add_argument('--interval', '-i', type=float, default=0.1)
    parser.add_argument('--data_size', '-s', type=int, default=1)
    parser.add_argument('--file', '-f', type=str, default="tmp_p1")
    args = parser.parse_args()
    Parallel.runner(n_parallel_workers=3, protocol="tcp", topology="mesh", auto_recover=True, max_retries=1)(
        SockTest.main, args.epoch, args.interval, args.data_size, args.file)
