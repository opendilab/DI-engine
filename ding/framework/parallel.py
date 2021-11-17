import atexit
import os
import time
import random
import pynng
import asyncio
import pickle
import logging
from os import path
from typing import Any, Callable, List, Optional
from threading import Thread

from mpire.pool import WorkerPool
from pynng.nng import Bus0, Socket
from rich import print

# Avoid ipc address conflict, random should always use random seed
random = random.Random()


class Parallel:

    def __init__(self, n_workers: int) -> None:
        self.n_workers = n_workers
        self._listener = None
        self._sock: Socket = None
        self._rpc = {"echo": self.echo}
        self._bind_addr = None

    def run(self, main_process: Callable, attach_to: List[str] = None) -> None:
        node_name = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=4))
        nodes = ["ipc:///tmp/ditask_{}_{}.ipc".format(node_name, i) for i in range(self.n_workers)]
        atexit.register(lambda: self.cleanup_nodes(nodes))
        attach_to = attach_to or []
        print("Bind subprocesses on these addresses: {}".format(nodes))

        def _parallel(node_id):
            self._listener = Thread(target=lambda: self.listen(node_id, nodes, attach_to), name="paralllel_listener")
            self._listener.start()
            time.sleep(0.5)  # Wait for thread start
            main_process()
            self._listener.join()

        with WorkerPool(n_jobs=self.n_workers) as pool:
            results = pool.map(_parallel, range(self.n_workers))
        return results

    def cleanup_nodes(self, nodes: List[str]) -> None:
        for node in nodes:
            ipc_file = node.split("//")[1]
            if path.exists(ipc_file):
                os.remove(ipc_file)

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
