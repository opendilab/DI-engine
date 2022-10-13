import time
from typing import Optional, Tuple, Any
from ding.utils import LockContext, LockContextType
from ditk import logging
from ding.utils import timeout_wrapper


class BarrierRuntime:

    def __init__(self, rank: int, world_size: int, debug: bool):
        """
        Overview:
            Barrier-related runtime states and the implementation of the state transition
            function.
        Arguments:
            - rank (int): [description]
            - world_size (int): [description]
            - debug (bool): To avoid confusion with user logs, we only print internal logs
                    when debug is explicitly specified.
        """
        self.rank = rank
        self.world_size = world_size
        self._debug = debug

        self.barrier_epoch = 0
        self.barrier_count = 0
        self.in_barrier = False
        self.barrier_count_list = [0 for i in range(self.world_size)]
        self.barrier_fin_count = 0
        self.barrier_lock = LockContext(LockContextType.THREAD_LOCK)

    def add_epoch(self):
        self.barrier_epoch += 1

    def reset_barrier_count(self):
        self.barrier_count = 0
        self.barrier_count_list = [0 for i in range(self.world_size)]
        self.barrier_fin_count = 0

    def reset_for_next_epoch(self):
        logging.warning(
            "timeout to meet target number: [{}] of processes when call barrier, now barrier count is {} retry..".
            format(self.world_size - 1, self.barrier_count)
        )
        self.barrier_count = 0
        self.add_epoch()

    def pickle_barrier_tag(self):
        # We use int to represent the tag of a barrier call, where the lower 2 decimal digits
        # represent its own Rank; the remaining digits represent the epoch when this barrier
        # method was called.
        return int(self.barrier_epoch * 100 + self.rank).to_bytes(32, byteorder='big')

    @staticmethod
    def unpickle_barrier_tag(payload):
        msg = int.from_bytes(payload, byteorder='big')
        return msg % 100, msg // 100

    def slave_barrier_prepare_step(self, payload):
        if self.rank > 0:
            if not self.in_barrier and self._debug:
                logging.debug("Node {} not call barrier yet, reject barrier resp".format(self.rank))
                return
            peer_rank, barrier_epoch = BarrierRuntime.unpickle_barrier_tag(payload)
            if peer_rank != 0:
                raise RuntimeError("The coordinator of barrier synchronization should be rank 0")
            self.barrier_epoch = barrier_epoch
            self.barrier_count += 1

            if self._debug:
                logging.debug(
                    "{} node recv peer_rank: {}, recv epoch:{}, barrier count {}".format(
                        self.rank, peer_rank, barrier_epoch, self.barrier_count
                    )
                )

    def master_barrier_prepare_step(self, payload):
        if self.rank == 0:
            peer_rank, epoch = BarrierRuntime.unpickle_barrier_tag(payload)
            if epoch < self.barrier_epoch:
                pass
            elif epoch > self.barrier_epoch:
                logging.warning(
                    "The epoch received by the coordinator is greater than the _barrier_epoch, \
                        check if multiple barrier calls are encountered."
                )
            else:
                self.barrier_count += 1
                self.barrier_count_list[peer_rank] = 1

    def slave_barrier_commit_step(self):
        if self.rank > 0:
            self.barrier_count = 0

    def master_barrier_commit_step(self):
        self.barrier_fin_count += 1


class BarrierContext:

    def __init__(self, runtime: BarrierRuntime):
        """
        Overview:
            A barrier function call context.
        Arguments:
            - runtime (BarrierRuntime): Barrier runtime
        """
        self._runtime = runtime

    def __enter__(self):
        if self._runtime.in_barrier:
            raise RuntimeError(
                "Unexpected behavior of the barrier, check if there has multiple barrier calls from multiple threads."
            )

        self._runtime.add_epoch()
        self._runtime.reset_barrier_count()
        self._runtime.in_barrier = True
        self._runtime.barrier_lock.acquire()
        if self._runtime._debug:
            logging.debug("Node {} barrier meeting begin".format(self._runtime.rank))

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            import traceback
            traceback.print_exception(exc_type, exc_value, tb)
        self._runtime.reset_barrier_count()
        self._runtime.in_barrier = False
        self._runtime.barrier_lock.release()
        if self._runtime._debug:
            logging.debug("Node {} barrier meeting finish".format(self._runtime.rank))


class MQ:
    """
    Overview:
        Abstract basic mq class.
    """

    def __init__(
            self,
            *args,
            rank: Optional[int] = 0,
            world_size: Optional[int] = 1,
            debug: Optional[bool] = False,
            **kwargs
    ) -> None:
        """
        Overview:
            The __init__ method of the inheritance must support the extra kwargs parameter.

            MQ can communicate with remote processes (tcp) and local processes (ipc),
            but cannot communicate with ipc and tcp at the same time.
        Arguments:
            - rank (Optional[int], optional): [description].
            - world_size (Optional[int], optional): [description].
            - debug (Optional[bool], optional): To avoid confusion with user logs, we only
                    print internal logs when debug is explicitly specified.
        """
        self.rank = rank
        self.world_size = world_size
        self._debug = debug
        # TODO(wangguoteng): make _rpc_topic from str to int
        self._rpc_topic = set(['s_pre', 'm_pre', 's_commit', 'm_commit'])
        self._barrier_ctx = BarrierRuntime(self.rank, self.world_size, self._debug)

    def listen(self) -> None:
        """
        Overview:
            Bind to local socket or connect to third party components.
        """
        raise NotImplementedError

    def publish(self, topic: str, data: bytes) -> None:
        """
        Overview:
            Send data to mq.
        Arguments:
            - topic (:obj:`str`): Topic.
            - data (:obj:`bytes`): Payload data.
        """
        raise NotImplementedError

    def subscribe(self, topic: str) -> None:
        """
        Overview:
            Subscribe to the topic.
        Arguments:
            - topic (:obj:`str`): Topic
        """
        raise NotImplementedError

    def unsubscribe(self, topic: str) -> None:
        """
        Overview:
            Unsubscribe from the topic.
        Arguments:
            - topic (:obj:`str`): Topic
        """
        raise NotImplementedError

    def recv(self) -> Tuple[str, bytes]:
        """
        Overview:
            Wait for incoming message, this function will block the current thread.
        Returns:
            - data (:obj:`Any`): The sent payload.
        """
        raise NotImplementedError

    def stop(self) -> None:
        """
        Overview:
            Unsubscribe from all topics and stop the connection to the message queue server.
        """
        return

    def _handle_private_topic(self, topic: str, payload: bytes) -> None:
        """
        Overview:
            Handler for MQ private topic not exposed to Parallel. Subclasses
            do not need to implement this method. But it should be noted that
            private rpc topics need to rely on Parallel's listening threads
            to do polling.

            Note that this method is not thread safe.
        Arguments:
            - topic (str): topic name
            - payload (bytes): payload
        """
        if topic == "s_pre":
            self._barrier_ctx.slave_barrier_prepare_step(payload)
        elif topic == "m_pre":
            self._barrier_ctx.master_barrier_prepare_step(payload)
        elif topic == "s_commit":
            self._barrier_ctx.slave_barrier_commit_step()
        elif topic == "m_commit":
            self._barrier_ctx.master_barrier_commit_step()
        else:
            logging.error("Unexpected rpc topic {}".format(topic))

    # TODO(wangguoteng): Specifies the scope of the process being synchronized.
    # should we move this method to BarrierRuntime?
    def barrier(self) -> None:
        """
            Synchronizes all RPC processes.

            This will block until all local or remote RPC processes reach this method
            to wait for all outstanding work to complete.

            A method similar to 2PC is used to achieve distributed processes
            synchronization.

            Subclasses do not need to implement this method.
        """

        def slave_prepare_loop_cond(ctx, old_count, *arg):
            return ctx.barrier_count == old_count

        def slave_commit_loop_cond(ctx, *arg):
            return ctx.barrier_count != 0

        def master_prepare_loop_cond(ctx, *arg):
            return sum(ctx.barrier_count_list) != ctx.world_size - 1

        def master_commit_loop_cond(ctx, *arg):
            return ctx.barrier_fin_count != ctx.world_size - 1

        def busy_loop(condition: callable, *arg, timeout: Optional[int] = None):

            @timeout_wrapper(timeout=timeout)
            def main_loop():
                if self._debug:
                    logging.debug("Node {} call: \"{}\"".format(self.rank, condition.__name__))
                while condition(self._barrier_ctx, *arg):
                    time.sleep(0.5)

            main_loop()

        if self.world_size == 1:
            return

        with BarrierContext(self._barrier_ctx):
            while True:
                if self.rank == 0:
                    try:
                        # Step 1: Send barrier prepare request to all other processes.
                        self.publish("s_pre", self._barrier_ctx.pickle_barrier_tag())

                        # Step 2: Wait for all other processes reply prepare-ack.
                        busy_loop(master_prepare_loop_cond, timeout=10)

                        # Step 3: Send barrier confirm request to all other processes.
                        self.publish("s_commit", b'_')

                        # Setp 4: Wait for all other processes reply commit-ACK.
                        # Confirm that all slave states have been reset to avoid consecutive
                        # barrier calls from the master matching the same slave's barrier call.
                        busy_loop(master_commit_loop_cond)
                    except TimeoutError as e:
                        # If there is a process that is not ready to connect (such as calling the
                        # barrier at the beginning of the program), our published message may be
                        # lost, so if the barrier's handshake times out, we will restart barrier
                        # handshake.
                        self._barrier_ctx.reset_for_next_epoch()
                        time.sleep(0.5)
                    except BaseException as e:
                        logging.error("Uncxcepted error: {}, barrier meeting failed!".format(e))
                        raise e
                    else:
                        self._barrier_ctx.reset_barrier_count()
                        break
                else:
                    try:
                        # Step 1: Wait for master (Node 0) to send the prepare request.
                        busy_loop(slave_prepare_loop_cond, self._barrier_ctx.barrier_count)

                        # Step 2: reply prepare-ACK to master here.
                        self.publish("m_pre", self._barrier_ctx.pickle_barrier_tag())

                        # Step 3: Wait for master to send barrier commit request.
                        busy_loop(slave_commit_loop_cond)

                        # Step 4: Reply commit-ACK ot master.
                        self.publish("m_commit", b'_')
                    except BaseException as e:
                        # We will not set the slave timeout so that the master can always
                        # achieve synchronization with them.
                        logging.debug("Node {}, RuntimeError: {}".format(self.rank, e))
                        raise e
                    else:
                        self._barrier_ctx.reset_barrier_count()
                        break
