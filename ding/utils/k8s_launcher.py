import yaml
import os
import subprocess
from enum import Enum, unique

@unique
class K8sType(Enum):
    Local = 1
    K3s = 2

class K8sLauncher(object):
    """
    Overview: object to manage the K8s cluster
    """
    def __init__(self, config_path: str) -> None:
        self.name = None
        self.servers = 1
        self.agents = 0
        self.type = K8sType.Local
        self.preload_images = []

        self._load(config_path)
        self._check_k3d_tools()

    def _load(self, config_path: str) -> None:
        with open(config_path, 'r') as f:
            data = yaml.load(f)
            for k, v in data.items():
                if k == 'name':
                    self.name = v
                elif k == 'servers':
                    self.servers = v
                elif k == 'agents':
                    self.agents = v
                elif k == 'type':
                    if v == 'k3s':
                        self.type = K8sType.K3s
                    elif v == 'local':
                        self.type = K8sType.Local
                    else:
                        raise ValueError("no type found for {}".format(v))
                elif k == 'preload_images':
                    self.preload_images = v
    
    def _check_k3d_tools(self) -> None:
        if self.type != K8sType.K3s:
            return
        args = ['which', 'k3d']
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, _ = proc.communicate()
        if out.decode('utf-8') == '':
            raise FileNotFoundError("No k3d tools found, please install k3d by executing ./hack/install-k3d.sh")

    def create_cluster(self) -> None:
        args = ['k3d', 'cluster', 'create', f'{self.name}', f'--servers={self.servers}', f'--agents={self.agents}']
        proc = subprocess.Popen(args, stderr=subprocess.PIPE)
        _, err = proc.communicate()
        if err.decode('utf-8') != '':
            raise RuntimeError(f'failed to create cluster {self.name}: {err.decode("utf8")}')

    def delete_cluster(self) -> None:
        args = ['k3d', 'cluster', 'delete', f'{self.name}']
        proc = subprocess.Popen(args, stderr=subprocess.PIPE)
        _, err = proc.communicate()
        if err.decode('utf-8') != '':
            raise RuntimeError(f'failed to delete cluster {self.name}: {err.decode("utf8")}')