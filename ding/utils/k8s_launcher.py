import yaml
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
        self._images = []

        self._load(config_path)
        self._check_k3d_tools()

    def _load(self, config_path: str) -> None:
        with open(config_path, 'r') as f:
            data = yaml.load(f)
            self.name = data.get('name') if data.get('name') else self.name
            if data.get('servers'):
                if type(data.get('servers')) is not int:
                    raise TypeError(f"servers' type is expected int, actual {type(data.get('servers'))}")
                self.servers = data.get('servers')
            if data.get('agents'):
                if type(data.get('agents')) is not int:
                    raise TypeError(f"agents' type is expected int, actual {type(data.get('agents'))}")
                self.agents = data.get('agents')
            if data.get('type'):
                if data.get('type') == 'k3s':
                    self.type = K8sType.K3s
                elif data.get('type') == 'local':
                    self.type = K8sType.Local
                else:
                    raise ValueError(f"no type found for {data.get('type')}")
            if data.get('preload_images'):
                if type(data.get('preload_images')) is not list:
                    raise TypeError(f"preload_images' type is expected list, actual {type(data.get('preload_images'))}")
                self._images = data.get('preload_images')

    def _check_k3d_tools(self) -> None:
        if self.type != K8sType.K3s:
            return
        args = ['which', 'k3d']
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, _ = proc.communicate()
        if out.decode('utf-8') == '':
            raise FileNotFoundError("No k3d tools found, please install by executing ./hack/install-k8s-tools.sh")

    def create_cluster(self) -> None:
        print('Creating k8s cluster...')
        if self.type != K8sType.K3s:
            return
        args = ['k3d', 'cluster', 'create', f'{self.name}', f'--servers={self.servers}', f'--agents={self.agents}']
        proc = subprocess.Popen(args, stderr=subprocess.PIPE)
        _, err = proc.communicate()
        err_str = err.decode('utf-8').strip()
        if err_str != '' and 'WARN' not in err_str:
            raise RuntimeError(f'Failed to create cluster {self.name}: {err_str}')

        # preload images
        self.preload_images(self._images)

    def delete_cluster(self) -> None:
        print('Deleting k8s cluster...')
        if self.type != K8sType.K3s:
            return
        args = ['k3d', 'cluster', 'delete', f'{self.name}']
        proc = subprocess.Popen(args, stderr=subprocess.PIPE)
        _, err = proc.communicate()
        err_str = err.decode('utf-8').strip()
        if err_str != '' and 'WARN' not in err_str and \
            'NotFound' not in err_str:
            raise RuntimeError(f'Failed to delete cluster {self.name}: {err_str}')

    def preload_images(self, images: list) -> None:
        if self.type != K8sType.K3s:
            return
        args = ['k3d', 'image', 'import', f'--cluster={self.name}']
        args += images

        proc = subprocess.Popen(args, stderr=subprocess.PIPE)
        _, err = proc.communicate()
        err_str = err.decode('utf-8').strip()
        if err_str != '' and 'WARN' not in err_str:
            raise RuntimeError(f'Failed to preload images: {err_str}')
