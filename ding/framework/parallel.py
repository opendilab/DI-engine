import atexit
import os
import random
import time
import traceback
from mpire.pool import WorkerPool
import pynng
import pickle
import logging
import tempfile
import socket
from os import path
from typing import Callable, Dict, List, Optional, Tuple, Union, Set
from threading import Thread
from pynng.nng import Bus0, Socket
from ding.framework.event_loop import EventLoop
from ding.utils.design_helper import SingletonMetaclass

# Avoid ipc address conflict, random should always use random seed
random = random.Random()


class Parallel(metaclass=SingletonMetaclass):

    def __init__(self) -> None:
        # Init will only be called once in a process
        self._listener = None
        self._sock: Socket = None
        self._bind_addr = None
        self.is_active = False
        self.attach_to = None
        self.finished = False
        self.node_id = None
        self.labels = set()
        self._event_loop = EventLoop("parallel_{}".format(id(self)))
        self._retries = 0  # Retries in auto recovery

    def run(
            self,
            node_id: int,
            listen_to: str,
            attach_to: Optional[List[str]] = None,
            labels: Optional[Set[str]] = None,
            auto_recover: bool = False,
            max_retries: int = float("inf")
    ) -> None:
        self.node_id = node_id
        self.attach_to = attach_to = attach_to or []
        self.labels = labels or set()
        self.auto_recover = auto_recover
        self.max_retries = max_retries
        self._listener = Thread(
            target=self.listen,
            kwargs={
                "listen_to": listen_to,
                "attach_to": attach_to
            },
            name="paralllel_listener",
            daemon=True
        )
        self._listener.start()

    @staticmethod
    def runner(
            n_parallel_workers: int,
            attach_to: Optional[List[str]] = None,
            protocol: str = "ipc",
            address: Optional[str] = None,
            ports: Optional[Union[List[int], int]] = None,
            topology: str = "mesh",
            labels: Optional[Set[str]] = None,
            node_ids: Optional[Union[List[int], int]] = None,
            auto_recover: bool = False,
            max_retries: int = float("inf")
    ) -> Callable:
        """
        Overview:
            This method allows you to configure parallel parameters, and now you are still in the parent process.
        Arguments:
            - n_parallel_workers (:obj:`int`): Workers to spawn.
            - attach_to (:obj:`Optional[List[str]]`): The node's addresses you want to attach to.
            - protocol (:obj:`str`): Network protocol.
            - address (:obj:`Optional[str]`): Bind address, ip or file path.
            - ports (:obj:`Optional[List[int]]`): Candidate ports.
            - topology (:obj:`str`): Network topology, includes:
                `mesh` (default): fully connected between each other;
                `star`: only connect to the first node;
                `alone`: do not connect to any node, except the node attached to;
            - labels (:obj:`Optional[Set[str]]`): Labels.
            - node_ids (:obj:`Optional[List[int]]`): Candidate node ids.
            - auto_recover (:obj:`bool`): Auto recover from uncaught exceptions from main.
            - max_retries (:obj:`int`): Max retries for auto recover.
        Returns:
            - _runner (:obj:`Callable`): The wrapper function for main.
        """
        attach_to = attach_to or []
        assert n_parallel_workers > 0, "Parallel worker number should bigger than 0"

        def _runner(main_process: Callable, *args, **kwargs) -> None:
            """
            Overview:
                Prepare to run in subprocess.
            Arguments:
                - main_process (:obj:`Callable`): The main function, your program start from here.
            """
            nodes = Parallel.get_node_addrs(n_parallel_workers, protocol=protocol, address=address, ports=ports)
            logging.warning("Bind subprocesses on these addresses: {}".format(nodes))

            def cleanup_nodes():
                for node in nodes:
                    protocol, file_path = node.split("://")
                    if protocol == "ipc" and path.exists(file_path):
                        os.remove(file_path)

            atexit.register(cleanup_nodes)

            def topology_network(i: int) -> List[str]:
                if topology == "mesh":
                    return nodes[:i] + attach_to
                elif topology == "star":
                    return nodes[:min(1, i)] + attach_to
                elif topology == "alone":
                    return attach_to
                else:
                    raise ValueError("Unknown topology: {}".format(topology))

            params_group = []
            candidate_node_ids = Parallel.padding_param(node_ids, n_parallel_workers, 0)
            assert len(candidate_node_ids) == n_parallel_workers, \
                "The number of workers must be the same as the number of node_ids, \
now there are {} workers and {} nodes"\
                    .format(n_parallel_workers, len(candidate_node_ids))
            for i in range(n_parallel_workers):
                runner_args = []
                runner_kwargs = {
                    "node_id": candidate_node_ids[i],
                    "listen_to": nodes[i],
                    "attach_to": topology_network(i),
                    "labels": labels,
                    "auto_recover": auto_recover,
                    "max_retries": max_retries
                }
                params = [(runner_args, runner_kwargs), (main_process, args, kwargs)]
                params_group.append(params)

            if n_parallel_workers == 1:
                Parallel.subprocess_runner(*params_group[0])
            else:
                with WorkerPool(n_jobs=n_parallel_workers, start_method="spawn", daemon=False) as pool:
                    # Cleanup the pool just in case the program crashes.
                    atexit.register(pool.__exit__)
                    pool.map(Parallel.subprocess_runner, params_group)

        return _runner

    @staticmethod
    def subprocess_runner(runner_params: Tuple[Union[List, Dict]], main_params: Tuple[Union[List, Dict]]) -> None:
        """
        Overview:
            Really run in subprocess.
        Arguments:
            - runner_params (:obj:`Tuple[Union[List, Dict]]`): Args and kwargs for runner.
            - main_params (:obj:`Tuple[Union[List, Dict]]`): Args and kwargs for main function.
        """
        main_process, args, kwargs = main_params
        runner_args, runner_kwargs = runner_params

        with Parallel() as router:
            router.is_active = True
            router.run(*runner_args, **runner_kwargs)
            time.sleep(0.3)  # Waiting for network pairing
            router.supervised_runner(main_process, *args, **kwargs)

    def supervised_runner(self, main: Callable, *args, **kwargs) -> None:
        """
        Overview:
            Run in supervised mode.
        Arguments:
            - main (:obj:`Callable`): Main function.
        """
        if self.auto_recover:
            while True:
                try:
                    main(*args, **kwargs)
                    break
                except Exception as e:
                    if self._retries < self.max_retries:
                        logging.warning(
                            "Auto recover from exception: {}, node: {}, retries: {}".format(
                                e, self.node_id, self._retries
                            )
                        )
                        logging.warning(traceback.format_exc())
                        self._retries += 1
                    else:
                        logging.warning(
                            "Exceed the max retries, node: {}, retries: {}, max_retries: {}".format(
                                self.node_id, self._retries, self.max_retries
                            )
                        )
                        raise e
        else:
            main(*args, **kwargs)

    @staticmethod
    def get_node_addrs(
            n_workers: int,
            protocol: str = "ipc",
            address: Optional[str] = None,
            ports: Optional[Union[List[int], int]] = None
    ) -> None:
        if protocol == "ipc":
            node_name = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=4))
            tmp_dir = tempfile.gettempdir()
            nodes = ["ipc://{}/ditask_{}_{}.ipc".format(tmp_dir, node_name, i) for i in range(n_workers)]
        elif protocol == "tcp":
            address = address or Parallel.get_ip()
            ports = Parallel.padding_param(ports, n_workers, 50515)
            assert len(ports) == n_workers, "The number of ports must be the same as the number of workers, \
now there are {} ports and {} workers".format(len(ports), n_workers)
            nodes = ["tcp://{}:{}".format(address, port) for port in ports]
        else:
            raise Exception("Unknown protocol {}".format(protocol))
        return nodes

    @staticmethod
    def padding_param(int_or_list: Optional[Union[List[int], int]], n_max: int, start_value: int) -> List[int]:
        """
        Overview:
            Padding int or list param to the length of n_max.
        Arguments:
            - int_or_list (:obj:`Optional[Union[List[int], int]]`): Int or list typed value.
            - n_max (:obj:`int`): Max length.
            - start_value (:obj:`int`): Start from value.
        """
        param = int_or_list
        if isinstance(param, List) and len(param) == 1:
            param = param[0]  # List with only 1 element is equal to int

        if isinstance(param, int):
            param = range(param, param + n_max)
        else:
            param = param or range(start_value, start_value + n_max)
        return param

    def listen(self, listen_to: str, attach_to: List[str] = None):
        attach_to = attach_to or []
        self._bind_addr = listen_to

        with Bus0() as sock:
            self._sock = sock
            sock.listen(self._bind_addr)
            time.sleep(0.1)  # Wait for peers to bind
            for contact in attach_to:
                sock.dial(contact)

            while True:
                try:
                    msg = sock.recv()
                    self._handle_message(msg)
                except pynng.Timeout:
                    logging.warning("Timeout on node {} when waiting for message from bus".format(self._bind_addr))
                except pynng.Closed:
                    if not self.finished:
                        logging.error("The socket was not closed under normal circumstances!")
                    break
                except Exception as e:
                    logging.error("Meet exception when listening for new messages", e)
                    break

    def on(self, event: str, fn: Callable) -> None:
        """
        Overview:
            Register an remote event on parallel instance, this function will be executed \
            when a remote process emit this event via network.
        Arguments:
            - event (:obj:`str`): Event name.
            - fn (:obj:`Callable`): Function body.
        """
        self._event_loop.on(event, fn)

    def once(self, event: str, fn: Callable) -> None:
        """
        Overview:
            Register an remote event which will only call once on parallel instance,
            this function will be executed when a remote process emit this event via network.
        Arguments:
            - event (:obj:`str`): Event name.
            - fn (:obj:`Callable`): Function body.
        """
        self._event_loop.once(event, fn)

    def off(self, event: str) -> None:
        """
        Overview:
            Unregister an event.
        Arguments:
            - event (:obj:`str`): Event name.
        """
        self._event_loop.off(event)

    def emit(self, event: str, *args, **kwargs) -> None:
        """
        Overview:
            Send an remote event via network to subscribed processes.
        Arguments:
            - event (:obj:`str`): Event name.
        """
        if self.is_active:
            topic = event + "::"
            payload = {"a": args, "k": kwargs}
            try:
                data = pickle.dumps(payload, protocol=-1)
            except AttributeError as e:
                logging.error("Arguments are not pickable! Event: {}, Args: {}".format(event, args))
                raise e
            data = topic.encode() + data
            return self._sock and self._sock.send(data)

    def _handle_message(self, msg: bytes) -> None:
        """
        Overview:
            Recv and parse payload from other processes, and call local functions.
        Arguments:
            - msg (:obj:`bytes`): Recevied message.
        """
        # Use topic at the beginning of the message, so we don't need to call pickle.loads
        # when the current process is not subscribed to the topic.
        topic, payload = msg.split(b"::", maxsplit=1)
        event = topic.decode()
        if not self._event_loop.listened(event):
            logging.debug("Event {} was not listened in parallel {}".format(event, self.node_id))
            return
        try:
            payload = pickle.loads(payload)
        except Exception as e:
            logging.error("Error when unpacking message on node {}, msg: {}".format(self._bind_addr, e))
            return
        self._event_loop.emit(event, *payload["a"], **payload["k"])

    @staticmethod
    def get_ip():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # doesn't even have to be reachable
            s.connect(('10.255.255.255', 1))
            IP = s.getsockname()[0]
        except Exception:
            IP = '127.0.0.1'
        finally:
            s.close()
        return IP

    def __enter__(self) -> "Parallel":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def stop(self):
        logging.info("Stopping parallel worker on address: {}".format(self._bind_addr))
        self.finished = True
        self.is_active = False
        time.sleep(0.03)
        if self._sock:
            self._sock.close()
        if self._listener:
            self._listener.join(timeout=1)
        self._event_loop.stop()
