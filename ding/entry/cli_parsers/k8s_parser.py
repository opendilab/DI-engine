import os
from typing import List, Optional


class K8SParser():

    def __init__(
            self, platform_spec: Optional[str] = None, topology: str = "alone", ports: str = None, **kwargs
    ) -> None:
        """
        Overview:
            Should only set global cluster properties
        """
        self.kwargs = kwargs
        self.nodelist = self._parse_node_list()
        self.ntasks = len(self.nodelist)
        self.platform_spec = platform_spec
        self.topology = topology
        self.ports = ports
        self.tasks = {}

    def parse(self) -> dict:
        procid = int(os.environ["K8S_PROCID"])
        nodename = self.nodelist[procid]
        task = self._get_task(procid)
        # Validation
        assert task["address"] == nodename
        return {**self.kwargs, **task}

    def _parse_node_list(self) -> List[str]:
        return os.environ["K8S_NODELIST"].split(",")

    def _get_task(self, procid: int) -> dict:
        """
        Overview:
            Complete node properties, use environment vars in list instead of on current node.
            For example, if you want to set nodename in this function, please derive it from K8S_NODELIST.
        Arguments:
            - procid (:obj:`int`): Proc order, starting from 0, must be set automatically by dijob.
                Note that it is different from node_id.
        """
        if procid in self.tasks:
            return self.tasks.get(procid)

        if self.platform_spec:
            task = self.platform_spec["tasks"][procid]
        else:
            task = {}
        if "ports" not in task:
            task["ports"] = self._get_ports()
        if "address" not in task:
            task["address"] = self._get_address(procid)
        if "node_ids" not in task:
            task["node_ids"] = procid

        task["attach_to"] = self._get_attach_to(procid, task.get("attach_to"))

        self.tasks[procid] = task
        return task

    def _get_attach_to(self, procid: int, attach_to: Optional[str] = None) -> str:
        """
        Overview:
            Parse from pattern of attach_to.
            If attach_to is specified in the platform_spec, it is formatted as a real address based on the specified address.
            If not, the real addresses will be generated based on the globally specified typology.
        Arguments:
            - procid (:obj:`int`): Proc order.
            - attach_to (:obj:`str`): The attach_to field in platform_spec for the task with current procid.
        Returns
            - attach_to (:obj:`str`): The real addresses for attach_to.
        """
        if attach_to:
            attach_to = [self._get_attach_to_part(part) for part in attach_to.split(",")]
        elif procid == 0:
            attach_to = []
        else:
            if self.topology == "mesh":
                prev_tasks = [self._get_task(i) for i in range(procid)]
                attach_to = [self._get_tcp_link(task["address"], task["ports"]) for task in prev_tasks]
            elif self.topology == "star":
                head_task = self._get_task(0)
                attach_to = self._get_tcp_link(head_task["address"], head_task["ports"])
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

    def _get_ports(self) -> str:
        return self.ports or "50515"

    def _get_address(self, procid: int) -> str:
        address = self.nodelist[procid]
        return address

    def _get_tcp_link(self, address: str, port: int) -> str:
        return "tcp://{}:{}".format(address, port)


def k8s_parser(platform_spec: Optional[str] = None, **kwargs) -> dict:
    return K8SParser(platform_spec, **kwargs).parse()
