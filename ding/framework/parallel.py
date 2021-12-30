import atexit
import os
import random
import time
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
from ding.utils.design_helper import SingletonMetaclass
from rich import print

# Avoid ipc address conflict, random should always use random seed
random = random.Random()


class Parallel(metaclass=SingletonMetaclass):

    def __init__(self) -> None:
        self._listener = None
        self._sock: Socket = None
        self._rpc = {}
        self._bind_addr = None
        self.is_active = False
        self.attach_to = None
        self.finished = False
        self.node_id = None
        self.labels = set()

    def run(
            self,
            node_id: int,
            listen_to: str,
            attach_to: Optional[List[str]] = None,
            labels: Optional[Set[str]] = None
    ) -> None:
        self.node_id = node_id
        self.attach_to = attach_to = attach_to or []
        self.labels = labels or set()
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
            ports: Optional[List[int]] = None,
            topology: str = "mesh",
            labels: Optional[Set[str]] = None,
            node_ids: Optional[List[int]] = None
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
            logging.info("Bind subprocesses on these addresses: {}".format(nodes))
            print("Bind subprocesses on these addresses: {}".format(nodes))

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
            candidate_node_ids = node_ids or range(n_parallel_workers)
            assert len(candidate_node_ids) == n_parallel_workers, \
                "The number of workers must be the same as the number of node_ids, \
now there are {} workers and {} nodes"\
                    .format(n_parallel_workers, len(candidate_node_ids))
            for i in range(n_parallel_workers):
                runner_args = []
                runner_kwargs = {
                    "node_id": candidate_node_ids[i],
                    "listen_to": nodes[i],
                    "attach_to": topology_network(i) + attach_to,
                    "labels": labels
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
            main_process(*args, **kwargs)

    @staticmethod
    def get_node_addrs(
            n_workers: int,
            protocol: str = "ipc",
            address: Optional[str] = None,
            ports: Optional[List[int]] = None
    ) -> None:
        if protocol == "ipc":
            node_name = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=4))
            tmp_dir = tempfile.gettempdir()
            nodes = ["ipc://{}/ditask_{}_{}.ipc".format(tmp_dir, node_name, i) for i in range(n_workers)]
        elif protocol == "tcp":
            address = address or Parallel.get_ip()
            ports = ports or range(50515, 50515 + n_workers)
            if isinstance(ports, int):
                ports = range(ports, ports + n_workers)
            assert len(ports) == n_workers, "The number of ports must be the same as the number of workers, \
now there are {} ports and {} workers".format(len(ports), n_workers)
            nodes = ["tcp://{}:{}".format(address, port) for port in ports]
        else:
            raise Exception("Unknown protocol {}".format(protocol))
        return nodes

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
                    msg = sock.recv_msg()
                    self.recv_rpc(msg.bytes)
                except pynng.Timeout:
                    logging.warning("Timeout on node {} when waiting for message from bus".format(self._bind_addr))
                except pynng.Closed:
                    if not self.finished:
                        logging.error("The socket was not closed under normal circumstances!")
                    break
                except Exception as e:
                    logging.error("Meet exception when listening for new messages", e)
                    break

    def register_rpc(self, fn_name: str, fn: Callable) -> None:
        """
        Overview:
            Register an rpc on parallel instance, this function will be executed \
            when a remote process call this function via network.
        Arguments:
            - fn_name (:obj:`str`): Function name.
            - fn (:obj:`Callable`): Function body.
        """
        self._rpc[fn_name] = fn

    def unregister_rpc(self, fn_name: str) -> None:
        """
        Overview:
            Unregister an rpc function.
        Arguments:
            - fn_name (:obj:`str`): Function name.
        """
        if fn_name in self._rpc:
            del self._rpc[fn_name]

    def send_rpc(self, func_name: str, *args, **kwargs) -> None:
        """
        Overview:
            Send an rpc via network to subscribed processes.
        Arguments:
            - fn_name (:obj:`str`): Function name.
        """
        if self.is_active:
            payload = {"f": func_name, "a": args, "k": kwargs}
            return self._sock and self._sock.send(pickle.dumps(payload, protocol=-1))

    def recv_rpc(self, msg: bytes):
        try:
            payload = pickle.loads(msg)
        except Exception as e:
            logging.warning("Error when unpacking message on node {}, msg: {}".format(self._bind_addr, e))
        if payload["f"] in self._rpc:
            fn = self._rpc[payload["f"]]
            fn(*payload["a"], **payload["k"])
        else:
            logging.warning("There was no function named {} in rpc table".format(payload["f"]))

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
        self._rpc.clear()
        time.sleep(0.03)
        if self._sock:
            self._sock.close()
        if self._listener:
            self._listener.join(timeout=1)
