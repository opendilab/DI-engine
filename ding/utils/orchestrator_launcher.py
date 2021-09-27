import subprocess
import time
from ding.utils import K8sLauncher
from .default_helper import one_time_warning


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

        self._namespace = 'di-system'
        self._webhook = 'di-webhook'
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
            raise FileNotFoundError(
                "No kubectl tools found, please install by executing ./ding/scripts/install-k8s-tools.sh"
            )

    def create_orchestrator(self) -> None:
        print('Creating orchestrator...')
        if self.cluster is not None:
            self.cluster.preload_images(self._images)

        # create and wait for cert-manager to be available
        create_components_from_config(self.cert_manager)
        wait_to_be_ready(self._cert_manager_namespace, self._cert_manager_webhook)

        # create and wait for di-orchestrator to be available
        create_components_from_config(self.installer)
        wait_to_be_ready(self._namespace, self._webhook)

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


def create_components_from_config(config: str) -> None:
    args = ['kubectl', 'create', '-f', f'{config}']
    proc = subprocess.Popen(args, stderr=subprocess.PIPE)
    _, err = proc.communicate()
    err_str = err.decode('utf-8').strip()
    if err_str != '' and 'WARN' not in err_str:
        if 'already exists' in err_str:
            print(f'Components already exists: {config}')
        else:
            raise RuntimeError(f'Failed to launch components: {err_str}')


def wait_to_be_ready(namespace: str, component: str, timeout: int = 120) -> None:
    try:
        from kubernetes import config, client, watch
    except ModuleNotFoundError:
        one_time_warning("You have not installed kubernetes package! Please try 'pip install DI-engine[k8s]'.")
        exit(-1)

    config.load_kube_config()
    appv1 = client.AppsV1Api()
    w = watch.Watch()
    for event in w.stream(appv1.list_namespaced_deployment, namespace, timeout_seconds=timeout):
        # print("Event: %s %s %s" % (event['type'], event['object'].kind, event['object'].metadata.name))
        if event['object'].metadata.name.startswith(component) and \
            event['object'].status.ready_replicas is not None and \
                event['object'].status.ready_replicas >= 1:
            print(f'component {component} is ready for serving')
            w.stop()
