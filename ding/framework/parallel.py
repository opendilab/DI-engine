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
from os import path
from typing import Any, Callable, List, Optional
from threading import Thread

from mpire.pool import WorkerPool
from pynng.nng import Bus0, Socket
from rich import print

# Avoid ipc address conflict, random should always use random seed
random = random.Random()


class Parallel:

    def __init__(self) -> None:
        self._listener = None
        self._sock: Socket = None
        self._rpc = {"echo": self.echo}
        self._bind_addr = None

    def run(
            self,
            main_process: Callable,
            n_workers: int,
            attach_to: List[str] = None,
            protocol: str = "ipc",
            address: Optional[str] = None,
            ports: Optional[List[int]] = None
    ) -> None:
        attach_to = attach_to or []

        nodes = self.get_node_addrs(n_workers, protocol=protocol, address=address, ports=ports)
        logging.info("Bind subprocesses on these addresses: {}".format(nodes))

        def _cleanup_nodes():
            for node in nodes:
                protocol, file_path = node.split("://")
                if protocol == "ipc" and path.exists(file_path):
                    os.remove(file_path)

        atexit.register(_cleanup_nodes)

        def _parallel(node_id):
            self._listener = Thread(
                target=lambda: self.listen(node_id, nodes, attach_to), name="paralllel_listener", daemon=True
            )
            self._listener.start()
            time.sleep(0.3)  # Wait for thread starting
            main_process()

        with WorkerPool(n_jobs=n_workers) as pool:
            # Cleanup the pool just in case the program crashes.
            atexit.register(lambda: pool.__exit__())
            results = pool.map(_parallel, range(n_workers))
        return results

    def get_node_addrs(
            self,
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
            address = address or self.get_ip()
            ports = ports or range(50515, 50515 + n_workers)
            assert len(ports) == n_workers, "The number of ports must be the same as the number of workers, \
now there are {} ports and {} workers".format(len(ports), n_workers)
            nodes = ["tcp://{}:{}".format(address, port) for port in ports]
        else:
            raise Exception("Unknown protocol {}".format(protocol))
        return nodes

    def listen(self, node_id: int, nodes: List[str], attach_to: List[str] = None):

        async def _listen():
            bind_addr = nodes[node_id]
            self._bind_addr = bind_addr
            dial_addrs = nodes[:node_id] + attach_to

            with Bus0() as sock:
                self._sock = sock
                sock.listen(bind_addr)
                await asyncio.sleep(.3)  # Wait for peers to bind
                for contact in dial_addrs:
                    sock.dial(contact)

                while True:
                    try:
                        msg = await sock.arecv_msg()
                        await self.recv_rpc(msg.bytes)
                    except pynng.Timeout:
                        logging.warning("Timeout on node {} when waiting for message from bus".format(self._bind_addr))

        asyncio.run(_listen())

    def echo(self, msg):
        """
        Overview:
            Simply print out the received message
        """
        print("Echo on node {}".format(self._bind_addr), msg)

    def register_rpc(self, fn_name: str, fn: Callable) -> None:
        self._rpc[fn_name] = fn

    def send_rpc(self, func_name: str, *args, **kwargs):
        payload = {"f": func_name, "a": args, "k": kwargs}
        return self._sock.send(pickle.dumps(payload))

    async def asend_rpc(self, func_name: str, *args, **kwargs):
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

    def get_ip(self):
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
