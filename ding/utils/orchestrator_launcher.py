import subprocess
import time
from ding.utils import K8sLauncher
from kubernetes import config, client, watch


class OrchestratorLauncher(object):
    """
    Overview: object to manage di-orchestrator in existing k8s cluster
    """

    def __init__(
            self,
            version: str,
            name: str = 'di-orchestrator',
            cluster: K8sLauncher = None,
            registry: str = 'diorchestrator',
            cert_manager_version: str = 'v1.3.1',
            cert_manager_registry: str = 'quay.io/jetstack'
    ) -> None:
        self.name = name
        self.version = version
        self.cluster = cluster
        self.registry = registry
        self.cert_manager_version = cert_manager_version
        self.cert_manager_registry = cert_manager_registry

        self._cert_manager_namespace = 'cert-manager'
        self._cert_manager_webhook = 'cert-manager-webhook'

        self.installer = 'https://raw.githubusercontent.com/opendilab/' + \
        f'DI-orchestrator/{self.version}/config/di-manager.yaml'
        self.cert_manager = 'https://github.com/jetstack/' + \
        f'cert-manager/releases/download/{self.cert_manager_version}/cert-manager.yaml'

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
        print('Creating orchestrator...')
        if self.cluster is not None:
            self.cluster.preload_images(self._images)

        for item in [self.cert_manager, self.installer]:
            if item is self.installer:
                watch_pod_events(self._cert_manager_namespace, self._cert_manager_webhook)
            args = ['kubectl', 'create', '-f', f'{item}']
            proc = subprocess.Popen(args, stderr=subprocess.PIPE)
            _, err = proc.communicate()
            err_str = err.decode('utf-8').strip()
            if err_str != '' and 'WARN' not in err_str:
                raise RuntimeError(f'Failed to launch di-orchestrator: {err_str}')

    def delete_orchestrator(self) -> None:
        print('Deleting orchestrator...')
        for item in [self.cert_manager, self.installer]:
            args = ['kubectl', 'delete', '-f', f'{item}']
            proc = subprocess.Popen(args, stderr=subprocess.PIPE)
            _, err = proc.communicate()
            err_str = err.decode('utf-8').strip()
            if err_str != '' and 'WARN' not in err_str and \
                'NotFound' not in err_str:
                raise RuntimeError(f'Failed to delete di-orchestrator: {err_str}')


def watch_pod_events(namespace: str, pod: str, timeout: int = 60, phase: str = "Running") -> None:
    config.load_kube_config()
    v1 = client.CoreV1Api()
    w = watch.Watch()
    for event in w.stream(v1.list_namespaced_pod, namespace, timeout_seconds=timeout):
        if event['object'].metadata.name.startswith(pod) and \
            event['object'].status.phase == phase:
            print(f'pod {pod} matched the desired phase: {phase}, sleep for a few seconds')
            time.sleep(5)
            w.stop()