import os
import re
from time import sleep
import numpy as np
from typing import Any, Dict, List, Optional


class SlurmParser():

    def __init__(self, platform_spec: Optional[Dict] = None, **kwargs) -> None:
        """
        Overview:
            Should only set global cluster properties
        """
        self.kwargs = kwargs
        self.ntasks = int(os.environ["SLURM_NTASKS"])
        self.platform_spec = platform_spec
        self.tasks = {}
        self.ntasks_per_node = int(os.environ["SLURM_NTASKS_PER_NODE"])
        self.nodelist = self._parse_node_list()
        self.ports = int(kwargs.get("ports") or 15151)
        self.parallel_workers = kwargs.get("parallel_workers") or 1
        self.topology = kwargs.get("topology") or "alone"

    def parse(self) -> dict:
        procid = int(os.environ["SLURM_PROCID"])
        task = self._get_task(procid)
        # Validation
        assert task["address"] == os.environ["SLURMD_NODENAME"]
        return {**self.kwargs, **task}

    def _get_task(self, procid: int) -> Dict[str, Any]:
        if procid in self.tasks:
            return self.tasks.get(procid)
        if self.platform_spec:
            task = self.platform_spec["tasks"][procid]
        else:
            task = {}
        if "ports" not in task:
            task["ports"] = self._get_ports(procid)
        if "address" not in task:
            task["address"] = self._get_address(procid)
        if "node_ids" not in task:
            task["node_ids"] = self._get_node_id(procid)

        task["attach_to"] = self._get_attach_to(procid, task.get("attach_to"))
        task["topology"] = self.topology
        task["parallel_workers"] = self.parallel_workers

        self.tasks[procid] = task
        return task

    def _parse_node_list(self) -> List[str]:
        nodelist = os.environ["SLURM_NODELIST"]
        result = re.match(r"(.*)?\[(.*)\]$", nodelist)
        if result:
            prefix, tails = result.groups()
            nodelist = []
            for tail in tails.split(","):
                if "-" in tail:
                    start, stop = tail.split("-")
                    for number in range(int(start), int(stop) + 1):
                        nodelist.append(prefix + str(number))
                else:
                    nodelist.append(prefix + tail)
        elif isinstance(nodelist, str):
            nodelist = [nodelist]
        if self.ntasks_per_node > 1:
            expand_nodelist = []  # Expand node for each task
            for node in nodelist:
                for _ in range(self.ntasks_per_node):
                    expand_nodelist.append(node)
            nodelist = expand_nodelist
        return nodelist

    def _get_attach_to(self, procid: int, attach_to: Optional[str] = None) -> str:
        if attach_to:
            attach_to = [self._get_attach_to_part(part) for part in attach_to.split(",")]
        elif procid == 0:
            attach_to = []
        else:
            if self.topology == "mesh":
                prev_tasks = [self._get_task(i) for i in range(procid)]
                attach_to = [self._get_attach_to_from_task(task) for task in prev_tasks]
                attach_to = list(np.concatenate(attach_to))
            elif self.topology == "star":
                head_task = self._get_task(0)
                attach_to = self._get_attach_to_from_task(head_task)
            else:
                attach_to = []

        return ",".join(attach_to)

    def _get_attach_to_part(self, attach_part: str) -> str:
        """
        Overview:
            Parse each part of attach_to.
        Arguments:
            - attach_part (:obj:`str`): The attach_to field with specific pattern, e.g. $node:0
        Returns
            - attach_to (:obj:`str`): The real address, e.g. tcp://SH-0:50000
        """
        if not attach_part.startswith("$node."):
            return attach_part
        attach_node_id = int(attach_part[6:])
        attach_task = self._get_task(self._get_procid_from_nodeid(attach_node_id))
        return self._get_tcp_link(attach_task["address"], attach_task["ports"])

    def _get_attach_to_from_task(self, task: dict) -> List[str]:
        """
        Overview:
            Get attach_to list from task, note that parallel_workers will affact the connected processes.
        Arguments:
            - task (:obj:`dict`): The task object.
        Returns
            - attach_to (:obj:`str`): The real address, e.g. tcp://SH-0:50000
        """
        port = task.get("ports")
        address = task.get("address")
        ports = [int(port) + i for i in range(self.parallel_workers)]
        attach_to = [self._get_tcp_link(address, port) for port in ports]
        return attach_to

    def _get_procid_from_nodeid(self, nodeid: int) -> int:
        procid = None
        for i in range(self.ntasks):
            task = self._get_task(i)
            if task["node_ids"] == nodeid:
                procid = i
                break
        if procid is None:
            raise Exception("Can not find procid from nodeid: {}".format(nodeid))
        return procid

    def _get_ports(self, procid) -> int:
        return self.ports + (procid % self.ntasks_per_node) * self.parallel_workers

    def _get_address(self, procid: int) -> str:
        address = self.nodelist[procid]
        return address

    def _get_node_id(self, procid: int) -> int:
        return procid * self.parallel_workers

    def _get_tcp_link(self, address: str, port: int) -> str:
        return "tcp://{}:{}".format(address, port)


def slurm_parser(platform_spec: str, **kwargs) -> dict:
    return SlurmParser(platform_spec, **kwargs).parse()
