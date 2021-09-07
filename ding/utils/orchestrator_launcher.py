import subprocess
from ding.utils import K8sLauncher

class OrchestratorLauncher(object):
    """
    Overview: object to manage di-orchestrator in existing k8s cluster
    """
    def __init__(self, version: str, name: str='di-orchestrator', cluster: K8sLauncher=None, registry: str='diorchestrator', 
        cert_manager_version: str='v1.3.1', cert_manager_registry: str='quay.io/jetstack') -> None:
        self.name = name
        self.version = version
        self.cluster = cluster
        self.registry = registry
        self.cert_manager_version = cert_manager_version
        self.cert_manager_registry = cert_manager_registry

        self.installer = f'https://raw.githubusercontent.com/opendilab/DI-orchestrator/{self.version}/config/di-manager.yaml'
        self.cert_manager = f'https://github.com/jetstack/cert-manager/releases/download/{self.cert_manager_version}/cert-manager.yaml'
        self._images = [
            f'{self.registry}/di-operator:{self.version}',
            f'{self.registry}/di-webhook:{self.version}',
            f'{self.registry}/di-server:{self.version}',
            f'{self.cert_manager_registry}/cert-manager-cainjector:{self.cert_manager_version}',
            f'{self.cert_manager_registry}/cert-manager-controller:{self.cert_manager_version}',
            f'{self.cert_manager_registry}/cert-manager-webhook:{self.cert_manager_version}',
        ]

        self._check_kubectl_tools()

    def _check_kubectl_tools(self) -> None:
        args = ['which', 'kubectl']
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, _ = proc.communicate()
        if out.decode('utf-8') == '':
            raise FileNotFoundError("No kubectl tools found, please install by executing ./hack/install-k8s-tools.sh")

    def create_orchestrator(self) -> None:
        print(f'Creating orchestrator...')
        if self.cluster is not None:
            self.cluster.preload_images(self._images)

        for item in [self.cert_manager, self.installer]:
            args = ['kubectl', 'create', '-f', f'{item}']
            proc = subprocess.Popen(args, stderr=subprocess.PIPE)
            _, err = proc.communicate()
            if err.decode('utf-8') != '':
                raise RuntimeError(f'Failed to launch di-orchestrator: {err.decode("utf8")}')

    def delete_orchestrator(self) -> None:
        print(f'Deleting orchestrator...')
        args = ['kubectl', 'delete', '-f', f'{self.installer}']
        proc = subprocess.Popen(args, stderr=subprocess.PIPE)
        _, err = proc.communicate()
        if err.decode('utf-8') != '':
            raise RuntimeError(f'Failed to launch di-orchestrator: {err.decode("utf8")}')