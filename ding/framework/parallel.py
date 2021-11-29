import atexit
import os
import time
import random
import pynng
import asyncio
import pickle
import logging
import tempfile
import socket
import multiprocessing as mp
from multiprocessing import Process
from os import path
from typing import Callable, List, Optional
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
        self.is_subprocess = False
        self.attach_to = None
        self.finished = False
        self._process_pool = []

    def run(self, listen_to: str, attach_to: List[str] = None) -> None:
        self.attach_to = attach_to = attach_to or []
        self._listener = Thread(
            target=self.listen,
            kwargs={
                "listen_to": listen_to,
                "attach_to": attach_to
            },
            name="paralllel_listener",
            daemon=False
        )
        self._listener.start()
        time.sleep(0.3)  # Wait for thread starting

    @staticmethod
    def runner(
            n_parallel_workers,
            attach_to: List[str] = None,
            protocol: str = "ipc",
            address: Optional[str] = None,
            ports: Optional[List[int]] = None
    ) -> Callable:
        """
        Overview:
            Config to run in subprocess.
        """
        attach_to = attach_to or []

        def _runner(main_process: Callable, *args, **kwargs) -> Callable:
            """
            Overview:
                Prepare to run in subprocess.
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

            process_pool = []
            if mp.get_start_method() != "spawn":
                mp.set_start_method("spawn")

            for node_id in range(n_parallel_workers):
                runner_args = []
                runner_kwargs = {"listen_to": nodes[node_id], "attach_to": nodes[:node_id] + attach_to}
                params = [(runner_args, runner_kwargs), (main_process, args, kwargs)]
                p = Process(target=Parallel.subprocess_runner, args=params)
                p.start()
                process_pool.append(p)

            def cleanup_processes():
                for p in process_pool:
                    p.close()

            atexit.register(cleanup_processes)

            for p in process_pool:
                p.join()

        return _runner

    @staticmethod
    def subprocess_runner(runner_params, main_params):
        """
        Overview:
            Really run in subprocess.
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
                            logging.error("The socket is not closed under normal circumstances!")
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
        self._sock.close()
        self._listener.join()
