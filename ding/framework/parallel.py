import atexit
import os
import random
import time
import traceback
from mpire.pool import WorkerPool
import pickle
from ditk import logging
import tempfile
import socket
from os import path
from typing import Callable, Dict, List, Optional, Tuple, Union, Set
from threading import Thread
from ding.framework.event_loop import EventLoop
from ding.utils.design_helper import SingletonMetaclass
from ding.framework.message_queue import *
from ding.utils.registry_factory import MQ_REGISTRY

# Avoid ipc address conflict, random should always use random seed
random = random.Random()


class Parallel(metaclass=SingletonMetaclass):

    def __init__(self) -> None:
        # Init will only be called once in a process
        self._listener = None
        self.is_active = False
        self.node_id = None
        self.labels = set()
        self._event_loop = EventLoop("parallel_{}".format(id(self)))
        self._retries = 0  # Retries in auto recovery

    def _run(
            self,
            node_id: int,
            labels: Optional[Set[str]] = None,
            auto_recover: bool = False,
            max_retries: int = float("inf"),
            mq_type: str = "nng",
            **kwargs
    ) -> None:
        self.node_id = node_id
        self.labels = labels or set()
        self.auto_recover = auto_recover
        self.max_retries = max_retries
        self._mq = MQ_REGISTRY.get(mq_type)(**kwargs)
        self._listener = Thread(target=self.listen, name="mq_listener", daemon=True)
        self._listener.start()

    @classmethod
    def runner(
            cls,
            n_parallel_workers: int,
            mq_type: str = "nng",
            attach_to: Optional[List[str]] = None,
            protocol: str = "ipc",
            address: Optional[str] = None,
            ports: Optional[Union[List[int], int]] = None,
            topology: str = "mesh",
            labels: Optional[Set[str]] = None,
            node_ids: Optional[Union[List[int], int]] = None,
            auto_recover: bool = False,
            max_retries: int = float("inf"),
            redis_host: Optional[str] = None,
            redis_port: Optional[int] = None
    ) -> Callable:
        """
        Overview:
            This method allows you to configure parallel parameters, and now you are still in the parent process.
        Arguments:
            - n_parallel_workers (:obj:`int`): Workers to spawn.
            - mq_type (:obj:`str`): Embedded message queue type, i.e. nng, redis.
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
            - redis_host (:obj:`str`): Redis server host.
            - redis_port (:obj:`int`): Redis server port.
        Returns:
            - _runner (:obj:`Callable`): The wrapper function for main.
        """
        all_args = locals()
        del all_args["cls"]
        args_parsers = {"nng": cls._nng_args_parser, "redis": cls._redis_args_parser}

        assert n_parallel_workers > 0, "Parallel worker number should bigger than 0"

        def _runner(main_process: Callable, *args, **kwargs) -> None:
            """
            Overview:
                Prepare to run in subprocess.
            Arguments:
                - main_process (:obj:`Callable`): The main function, your program start from here.
            """
            runner_params = args_parsers[mq_type](**all_args)
            params_group = [[runner_kwargs, (main_process, args, kwargs)] for runner_kwargs in runner_params]

            if n_parallel_workers == 1:
                cls._subprocess_runner(*params_group[0])
            else:
                with WorkerPool(n_jobs=n_parallel_workers, start_method="spawn", daemon=False) as pool:
                    # Cleanup the pool just in case the program crashes.
                    atexit.register(pool.__exit__)
                    pool.map(cls._subprocess_runner, params_group)

        return _runner

    @classmethod
    def _nng_args_parser(
            cls,
            n_parallel_workers: int,
            attach_to: Optional[List[str]] = None,
            protocol: str = "ipc",
            address: Optional[str] = None,
            ports: Optional[Union[List[int], int]] = None,
            topology: str = "mesh",
            node_ids: Optional[Union[List[int], int]] = None,
            **kwargs
    ) -> Dict[str, dict]:
        attach_to = attach_to or []
        nodes = cls.get_node_addrs(n_parallel_workers, protocol=protocol, address=address, ports=ports)
        logging.info("Bind subprocesses on these addresses: {}".format(nodes))

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

        runner_params = []
        candidate_node_ids = cls.padding_param(node_ids, n_parallel_workers, 0)
        for i in range(n_parallel_workers):
            runner_kwargs = {
                **kwargs,
                "node_id": candidate_node_ids[i],
                "listen_to": nodes[i],
                "attach_to": topology_network(i),
            }
            runner_params.append(runner_kwargs)

        return runner_params

    @classmethod
    def _redis_args_parser(cls, n_parallel_workers: int, node_ids: Optional[Union[List[int], int]] = None, **kwargs):
        runner_params = []
        candidate_node_ids = cls.padding_param(node_ids, n_parallel_workers, 0)
        for i in range(n_parallel_workers):
            runner_kwargs = {**kwargs, "node_id": candidate_node_ids[i]}
            runner_params.append(runner_kwargs)
        return runner_params

    @classmethod
    def _subprocess_runner(cls, runner_kwargs: dict, main_params: Tuple[Union[List, Dict]]) -> None:
        """
        Overview:
            Really run in subprocess.
        Arguments:
            - runner_params (:obj:`Tuple[Union[List, Dict]]`): Args and kwargs for runner.
            - main_params (:obj:`Tuple[Union[List, Dict]]`): Args and kwargs for main function.
        """
        main_process, args, kwargs = main_params

        with Parallel() as router:
            router.is_active = True
            router._run(**runner_kwargs)
            time.sleep(0.3)  # Waiting for network pairing
            router._supervised_runner(main_process, *args, **kwargs)

    def _supervised_runner(self, main: Callable, *args, **kwargs) -> None:
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

    @classmethod
    def get_node_addrs(
            cls,
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
            address = address or cls.get_ip()
            ports = cls.padding_param(ports, n_workers, 50515)
            assert len(ports) == n_workers, "The number of ports must be the same as the number of workers, \
now there are {} ports and {} workers".format(len(ports), n_workers)
            nodes = ["tcp://{}:{}".format(address, port) for port in ports]
        else:
            raise Exception("Unknown protocol {}".format(protocol))
        return nodes

    @classmethod
    def padding_param(cls, int_or_list: Optional[Union[List[int], int]], n_max: int, start_value: int) -> List[int]:
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

    def listen(self):
        self._mq.listen()
        while True:
            if not self._mq:
                break
            msg = self._mq.recv()
            # msg is none means that the message queue is no longer being listened to,
            # especially if the message queue is already closed
            if not msg:
                break
            topic, msg = msg
            self._handle_message(topic, msg)

    def on(self, event: str, fn: Callable) -> None:
        """
        Overview:
            Register an remote event on parallel instance, this function will be executed \
            when a remote process emit this event via network.
        Arguments:
            - event (:obj:`str`): Event name.
            - fn (:obj:`Callable`): Function body.
        """
        if self.is_active:
            self._mq.subscribe(event)
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
        if self.is_active:
            self._mq.subscribe(event)
        self._event_loop.once(event, fn)

    def off(self, event: str) -> None:
        """
        Overview:
            Unregister an event.
        Arguments:
            - event (:obj:`str`): Event name.
        """
        if self.is_active:
            self._mq.unsubscribe(event)
        self._event_loop.off(event)

    def emit(self, event: str, *args, **kwargs) -> None:
        """
        Overview:
            Send an remote event via network to subscribed processes.
        Arguments:
            - event (:obj:`str`): Event name.
        """
        if self.is_active:
            payload = {"a": args, "k": kwargs}
            try:
                data = pickle.dumps(payload, protocol=-1)
            except AttributeError as e:
                logging.error("Arguments are not pickable! Event: {}, Args: {}".format(event, args))
                raise e
            self._mq.publish(event, data)

    def _handle_message(self, topic: str, msg: bytes) -> None:
        """
        Overview:
            Recv and parse payload from other processes, and call local functions.
        Arguments:
            - topic (:obj:`str`): Recevied topic.
            - msg (:obj:`bytes`): Recevied message.
        """
        event = topic
        if not self._event_loop.listened(event):
            logging.debug("Event {} was not listened in parallel {}".format(event, self.node_id))
            return
        try:
            payload = pickle.loads(msg)
        except Exception as e:
            logging.error("Error when unpacking message on node {}, msg: {}".format(self.node_id, e))
            return
        self._event_loop.emit(event, *payload["a"], **payload["k"])

    @classmethod
    def get_ip(cls):
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
        logging.info("Stopping parallel worker on node: {}".format(self.node_id))
        self.is_active = False
        time.sleep(0.03)
        if self._mq:
            self._mq.stop()
            self._mq = None
        if self._listener:
            self._listener.join(timeout=1)
            self._listener = None
        self._event_loop.stop()
