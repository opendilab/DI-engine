import os
import re
from typing import List


class SlurmParser():

    def __init__(self, platform_spec: str, **kwargs) -> None:
        """
        Overview:
            Should only set global cluster properties
        """
        self.kwargs = kwargs
        self.ntasks = int(os.environ["SLURM_NTASKS"])
        self.tasks = platform_spec["tasks"]
        self.ntasks_per_node = int(os.environ["SLURM_NTASKS_PER_NODE"])
        self.nodelist = self._parse_node_list()

    def parse(self) -> dict:
        assert len(self.tasks) == self.ntasks
        procid = int(os.environ["SLURM_PROCID"])
        nodename = os.environ["SLURMD_NODENAME"]
        task = self._get_node_args(procid)
        # Validation
        assert task["address"] == nodename
        return {**self.kwargs, **task}

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
        return nodelist

    def _get_node_args(self, procid: int) -> dict:
        """
        Overview:
            Complete node properties, use environment vars in list instead of on current node.
            For example, if you want to set nodename in this function, please derive it from SLURM_NODELIST,
            the variable from SLURMD_NODENAME should only be used in validation.
        """
        task = self.tasks[procid]
        if "attach_to" in task:
            task["attach_to"] = self._get_attach_to(task["attach_to"])
        if "address" not in task:
            task["address"] = self._get_address(procid)
        if "ports" not in task:
            task["ports"] = self._get_ports(procid)
        if "node_ids" not in task:
            task["node_ids"] = procid
        return task

    def _get_attach_to(self, attach_to: str) -> str:
        attach_to = [self._get_attach_to_part(part) for part in attach_to.split(",")]
        return ",".join(attach_to)

    def _get_attach_to_part(self, attach_part: str) -> str:
        if not attach_part.startswith("$node."):
            return attach_part
        attach_node_id = int(attach_part[6:])
        attach_node = self._get_node_args(self._get_procid_from_nodeid(attach_node_id))
        return "tcp://{}:{}".format(attach_node["address"], attach_node["ports"])

    def _get_procid_from_nodeid(self, nodeid: int) -> int:
        procid = None
        for i, task in enumerate(self.tasks):
            if task.get("node_ids") == nodeid:
                procid = i
                break
            elif nodeid == i:
                procid = i
                break
        if procid is None:
            raise Exception("Can not find procid from nodeid: {}".format(nodeid))
        return procid

    def _get_ports(self, procid: int) -> List[int]:
        ports = 15151 + procid % self.ntasks_per_node
        return ports

    def _get_address(self, procid: int) -> str:
        address = self.nodelist[procid // self.ntasks_per_node]
        return address


def slurm_parser(platform_spec: str, **kwargs) -> dict:
    return SlurmParser(platform_spec, **kwargs).parse()
