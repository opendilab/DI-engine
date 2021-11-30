import atexit
import os
import random
import threading
import time
from mpire.pool import WorkerPool
import pynng
import asyncio
import pickle
import logging
import tempfile
import socket
from os import path
from typing import Callable, Dict, List, Optional, Tuple, Union
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
        self._rpc = {"echo": self.echo}
        self._bind_addr = None
        self._lock = threading.Lock()
        self.is_subprocess = False
        self.attach_to = None
        self.finished = False

    def run(self, node_id: int, listen_to: str, attach_to: List[str] = None) -> None:
        self.node_id = node_id
        self.attach_to = attach_to = attach_to or []
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
            ports: Optional[List[int]] = None
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
        Returns:
            - _runner (:obj:`Callable`): The wrapper function for main.
        """
        attach_to = attach_to or []

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

            params_group = []
            for node_id in range(n_parallel_workers):
                runner_args = []
                runner_kwargs = {
                    "node_id": node_id,
                    "listen_to": nodes[node_id],
                    "attach_to": nodes[:node_id] + attach_to
                }
                params = [(runner_args, runner_kwargs), (main_process, args, kwargs)]
                params_group.append(params)

            with WorkerPool(n_jobs=n_parallel_workers, start_method="spawn") as pool:
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

        router = Parallel()
        router.is_subprocess = True
        router.run(*runner_args, **runner_kwargs)
        main_process(*args, **kwargs)
        router.stop()

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
            assert len(ports) == n_workers, "The number of ports must be the same as the number of workers, \
now there are {} ports and {} workers".format(len(ports), n_workers)
            nodes = ["tcp://{}:{}".format(address, port) for port in ports]
        else:
            raise Exception("Unknown protocol {}".format(protocol))
        return nodes

    def listen(self, listen_to: str, attach_to: List[str] = None):
        attach_to = attach_to or []

        async def _listen():
            self._bind_addr = listen_to

            with Bus0() as sock:
                self._sock = sock
                sock.listen(self._bind_addr)
                await asyncio.sleep(.3)  # Wait for peers to bind
                for contact in attach_to:
                    sock.dial(contact)

                while True:
                    try:
                        msg = await sock.arecv_msg()
                        await self.recv_rpc(msg.bytes)
                    except pynng.Timeout:
                        logging.warning("Timeout on node {} when waiting for message from bus".format(self._bind_addr))
                    except pynng.Closed:
                        if not self.finished:
                            logging.error("The socket was not closed under normal circumstances!")
                        break
                    except Exception as e:
                        logging.error("Meet exception when listening for new messages", e)
                        break

        asyncio.run(_listen())

    def echo(self, msg):
        """
        Overview:
            Simply print out the received message
        """
        print("Echo on node {}".format(self._bind_addr), msg)

    def register_rpc(self, fn_name: str, fn: Callable) -> None:
        self._rpc[fn_name] = fn

    def send_rpc(self, func_name: str, *args, **kwargs) -> None:
        if self.is_subprocess:
            payload = {"f": func_name, "a": args, "k": kwargs}
            return self._sock and self._sock.send(pickle.dumps(payload))

    async def asend_rpc(self, func_name: str, *args, **kwargs) -> None:
        if self.is_subprocess:
            msg = {"f": func_name, "a": args, "k": kwargs}
            return await self._sock.asend(pickle.dumps(msg))

    async def recv_rpc(self, msg: bytes):
        try:
            payload = pickle.loads(msg)
        except Exception as e:
            logging.warning("Error when unpacking message on node {}, msg: {}".format(self._bind_addr, e))
        if payload["f"] in self._rpc:
            self._rpc[payload["f"]](*payload["a"], **payload["k"])
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

    def stop(self):
        logging.info("Stopping parallel worker on address: {}".format(self._bind_addr))
        self.finished = True
        time.sleep(0.03)
        self._sock.close()
        self._listener.join(timeout=1)
