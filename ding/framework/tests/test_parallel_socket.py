import time
import logging
import argparse
from ding.framework import Parallel


class AutoRecoverV2:

    @classmethod
    def main_p0(cls, epoch, interval):
        router = Parallel()
        greets = []
        router.on("greeting_0", lambda msg: greets.append(msg))
        start_t = time.time()
        logging.info("main_p0 start ...")

        for i in range(epoch):
            while time.time() - start_t < i * interval:
                time.sleep(0.1)

            if not greets or i % 10 != 0:
                continue
            last_msg = greets[-1]
            msg_idx, msg_t = last_msg.split("_")[-2:]
            logging.info("main_p0 passed {:.2f} s, received {} msgs. last msg: idx {}, time {} s"\
                .format(time.time() - start_t, len(greets), msg_idx, msg_t))

        logging.info("main_p0 done! total msg: {}".format(len(greets)))

    @classmethod
    def main_p1(cls, file, epoch, interval):
        s = "rfvbhukm" * 1024 * 128
        print("msg length:", len(s.encode()))

        router = Parallel()
        greets = []
        router.on("greeting_1", lambda msg: greets.append(msg))
        start_t = time.time()
        logging.info("main_p1 start ...")

        with open(file, "w") as f:
            f.write("{}\n".format(router.get_ip()))

        for i in range(epoch):
            while time.time() - start_t < i * interval:
                time.sleep(0.1)

            if router._retries == 0:
                router.emit("greeting_0", "{}_{}_{:.2f}".format(s, i, time.time() - start_t))
            elif router._retries == 1:
                router.emit("greeting_0", "recovered_{}_{:.2f}".format(i, time.time() - start_t))
            else:
                raise Exception("Failed too many times")

            if not greets or i % 10 != 0:
                continue
            last_msg = greets[-1]
            msg_idx, msg_t = last_msg.split("_")[-2:]
            logging.info("main_p1 passed {:.2f} s, received {} msgs. last msg: idx {}, time {} s"\
                .format(time.time() - start_t, len(greets), msg_idx, msg_t))

        logging.info("main_p1 done! total msg: {} retries: {}".format(len(greets), router._retries))

    @classmethod
    def main_p2(cls, epoch, interval):
        s = "cnhudofs" * 1024 * 128
        print("msg length:", len(s.encode()))

        router = Parallel()
        start_t = time.time()
        logging.info("main_p2 start ...")

        for i in range(epoch):
            while time.time() - start_t < i * interval:
                time.sleep(0.1)

            router.emit("greeting_1", "{}_{}_{:.2f}".format(s, i, time.time() - start_t))

        logging.info("main_p2 done!")

    @classmethod
    def main(cls, file, epoch=1000, interval=1.0):
        router = Parallel()
        if router.node_id == 0:
            cls.main_p0(epoch, interval)
        elif router.node_id == 1:
            cls.main_p1(file, epoch, interval)
        elif router.node_id == 2:
            cls.main_p2(epoch, interval)
        else:
            raise Exception("Invalid node id")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', type=str, default="tmp_123")
    parser.add_argument('--epoch', '-t', type=int, default=1200)
    parser.add_argument('--interval', '-i', type=float, default=0.1)
    args = parser.parse_args()
    Parallel.runner(n_parallel_workers=3, protocol="tcp", topology="mesh", auto_recover=True, max_retries=1)(
        AutoRecoverV2.main, args.file, args.epoch, args.interval)


