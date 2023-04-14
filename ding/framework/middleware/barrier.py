from time import sleep, time
from ditk import logging
from ding.framework import task
from ding.utils.lock_helper import LockContext, LockContextType
from ding.utils.design_helper import SingletonMetaclass


class BarrierRuntime(metaclass=SingletonMetaclass):

    def __init__(self, node_id: int, max_world_size: int = 100):
        """
        Overview:
            'BarrierRuntime' is a singleton class. In addition, it must be initialized before the
            class 'Parallel' starts MQ, otherwise the messages sent by other nodes may be lost after
            the detection is completed. We don't have a message retransmission mechanism, and losing
            a message means deadlock.
        Arguments:
            - node_id (int): Process ID.
            - max_world_size (int, optional): The maximum total number of processes that can be
                synchronized, the defalut value is 100.
        """
        self.node_id = node_id
        self._has_detected = False
        self._range_len = len(str(max_world_size)) + 1

        self._barrier_epoch = 0
        self._barrier_recv_peers_buff = dict()
        self._barrier_recv_peers = dict()
        self._barrier_ack_peers = []
        self._barrier_lock = LockContext(LockContextType.THREAD_LOCK)

        self.mq_type = task.router.mq_type
        self._connected_peers = dict()
        self._connected_peers_lock = LockContext(LockContextType.THREAD_LOCK)
        self._keep_alive_daemon = False

        self._event_name_detect = "b_det"
        self.event_name_req = "b_req"
        self.event_name_ack = "b_ack"

    def _alive_msg_handler(self, peer_id):
        with self._connected_peers_lock:
            self._connected_peers[peer_id] = time()

    def _add_barrier_req(self, msg):
        peer, epoch = self._unpickle_barrier_tag(msg)
        logging.debug("Node:[{}] recv barrier request from node:{}, epoch:{}".format(self.node_id, peer, epoch))
        with self._barrier_lock:
            if peer not in self._barrier_recv_peers:
                self._barrier_recv_peers[peer] = []
            self._barrier_recv_peers[peer].append(epoch)

    def _add_barrier_ack(self, peer):
        logging.debug("Node:[{}] recv barrier ack from node:{}".format(self.node_id, peer))
        with self._barrier_lock:
            self._barrier_ack_peers.append(peer)

    def _unpickle_barrier_tag(self, msg):
        return msg % self._range_len, msg // self._range_len

    def pickle_barrier_tag(self):
        return int(self._barrier_epoch * self._range_len + self.node_id)

    def reset_all_peers(self):
        with self._barrier_lock:
            for peer, q in self._barrier_recv_peers.items():
                if len(q) != 0:
                    assert q.pop(0) == self._barrier_epoch
            self._barrier_ack_peers = []
            self._barrier_epoch += 1

    def get_recv_num(self):
        count = 0
        with self._barrier_lock:
            if len(self._barrier_recv_peers) > 0:
                for _, q in self._barrier_recv_peers.items():
                    if len(q) > 0 and q[0] == self._barrier_epoch:
                        count += 1
        return count

    def get_ack_num(self):
        with self._barrier_lock:
            return len(self._barrier_ack_peers)

    def detect_alive(self, expected, timeout):
        # The barrier can only block other nodes within the visible range of the current node.
        # If the 'attch_to' list of a node is empty, it does not know how many nodes will attach to him,
        # so we cannot specify the effective range of a barrier in advance.
        assert task._running
        task.on(self._event_name_detect, self._alive_msg_handler)
        task.on(self.event_name_req, self._add_barrier_req)
        task.on(self.event_name_ack, self._add_barrier_ack)
        start = time()
        while True:
            sleep(0.1)
            task.emit(self._event_name_detect, self.node_id, only_remote=True)
            # In case the other node has not had time to receive our detect message,
            # we will send an additional round.
            if self._has_detected:
                break
            with self._connected_peers_lock:
                if len(self._connected_peers) == expected:
                    self._has_detected = True

            if time() - start > timeout:
                raise TimeoutError("Node-[{}] timeout when waiting barrier! ".format(task.router.node_id))

        task.off(self._event_name_detect)
        logging.info(
            "Barrier detect node done, node-[{}] has connected with {} active nodes!".format(self.node_id, expected)
        )


class BarrierContext:

    def __init__(self, runtime: BarrierRuntime, detect_timeout, expected_peer_num: int = 0):
        self._runtime = runtime
        self._expected_peer_num = expected_peer_num
        self._timeout = detect_timeout

    def __enter__(self):
        if not self._runtime._has_detected:
            self._runtime.detect_alive(self._expected_peer_num, self._timeout)

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            import traceback
            traceback.print_exception(exc_type, exc_value, tb)
        self._runtime.reset_all_peers()


class Barrier:

    def __init__(self, attch_from_nums: int, timeout: int = 60):
        """
        Overview:
            Barrier() is a middleware for debug or profiling. It can synchronize the task step of each
            process within the scope of all visible processes. When using Barrier(), you need to pay
            attention to the following points:

            1. All processes must call the same number of Barrier(), otherwise a deadlock occurs.

            2. 'attch_from_nums' is a very important variable, This value indicates the number of times
                the current process will be attached to by other processes (the number of connections
                established).
                For example:
                    Node0: address: 127.0.0.1:12345, attach_to = []
                    Node1: address: 127.0.0.1:12346, attach_to = ["tcp://127.0.0.1:12345"]
                    For Node0, the 'attch_from_nums' value is 1. (It will be acttched by Node1)
                    For Node1, the 'attch_from_nums' value is 0. (No one will attach to Node1)
                Please note that this value must be given correctly, otherwise, for a node whose 'attach_to'
                list is empty, it cannot perceive how many processes will establish connections with it,
                resulting in any form of synchronization cannot be performed.

            3. Barrier() is thread-safe, but it is not recommended to use barrier in multithreading. You need
                to carefully calculate the number of times each thread calls Barrier() to avoid deadlock.

            4. In normal training tasks, please do not use Barrier(), which will force the step synchronization
                between each process, so it will greatly damage the training efficiency. In addition, if your
                training task has dynamic processes, do not use Barrier() to prevent deadlock.

        Arguments:
            - attch_from_nums (int): [description]
            - timeout (int, optional): The timeout for successful detection of 'expected_peer_num'
                number of nodes, the default value is 60 seconds.
        """
        self.node_id = task.router.node_id
        self.timeout = timeout
        self._runtime: BarrierRuntime = task.router.barrier_runtime
        self._barrier_peers_nums = task.get_attch_to_len() + attch_from_nums

        logging.info(
            "Node:[{}], attach to num is:{}, attach from num is:{}".format(
                self.node_id, task.get_attch_to_len(), attch_from_nums
            )
        )

    def __call__(self, ctx):
        self._wait_barrier(ctx)
        yield
        self._wait_barrier(ctx)

    def _wait_barrier(self, ctx):
        self_ready = False
        with BarrierContext(self._runtime, self.timeout, self._barrier_peers_nums):
            logging.debug("Node:[{}] enter barrier".format(self.node_id))
            # Step1: Notifies all the attached nodes that we have reached the barrier.
            task.emit(self._runtime.event_name_req, self._runtime.pickle_barrier_tag(), only_remote=True)
            logging.debug("Node:[{}] sended barrier request".format(self.node_id))

            # Step2: We check the number of flags we have received.
            # In the current CI design of DI-engine, there will always be a node whose 'attach_to' list is empty,
            # so there will always be a node that will send ACK unconditionally, so deadlock will not occur.
            if self._runtime.get_recv_num() == self._barrier_peers_nums:
                self_ready = True

            # Step3: Waiting for our own to be ready.
            # Even if the current process has reached the barrier, we will not send an ack immediately,
            # we need to wait for the slowest directly connected or indirectly connected peer to
            # reach the barrier.
            start = time()
            if not self_ready:
                while True:
                    if time() - start > self.timeout:
                        raise TimeoutError("Node-[{}] timeout when waiting barrier! ".format(task.router.node_id))

                    if self._runtime.get_recv_num() != self._barrier_peers_nums:
                        sleep(0.1)
                    else:
                        break

            # Step4: Notifies all attached nodes that we are ready.
            task.emit(self._runtime.event_name_ack, self.node_id, only_remote=True)
            logging.debug("Node:[{}] sended barrier ack".format(self.node_id))

            # Step5: Wait until all directly or indirectly connected nodes are ready.
            start = time()
            while True:
                if time() - start > self.timeout:
                    raise TimeoutError("Node-[{}] timeout when waiting barrier! ".format(task.router.node_id))

                if self._runtime.get_ack_num() != self._barrier_peers_nums:
                    sleep(0.1)
                else:
                    break

            logging.info("Node-[{}] env_step:[{}] barrier finish".format(self.node_id, ctx.env_step))
