import pytest
import os
from ding.utils import K8sLauncher, OrchestratorLauncher
from kubernetes import config, client


@pytest.mark.unittest
def test_create_and_delete_k8s_cluster():
    cluster_name = 'test-k8s-launcher'
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'k8s-config.yaml')
    launcher = K8sLauncher(config_path)
    launcher.name = cluster_name

    # create cluster
    launcher.create_cluster()

    # check that cluster is successfully created
    config.load_kube_config()
    current_context = config.list_kube_config_contexts()[1]
    assert current_context['context']['cluster'].startswith(f"k3d-{cluster_name}")

    # create orchestrator
    olauncher = OrchestratorLauncher('v0.2.0-rc.0', cluster=launcher)
    olauncher.create_orchestrator()

    # check orchestrator is successfully created
    appv1 = client.AppsV1Api()
    ret = appv1.list_namespaced_deployment("di-system")
    assert len(ret.items) == 3

    # check crds are installed
    extensionv1 = client.ApiextensionsV1Api()
    ret = extensionv1.list_custom_resource_definition()
    found = 0
    for crd in ret.items:
        found = found + 1 if crd.metadata.name == 'aggregatorconfigs.diengine.opendilab.org' else found
        found = found + 1 if crd.metadata.name == 'dijobs.diengine.opendilab.org' else found
    assert found == 2

    # delete orchestrator
    olauncher.delete_orchestrator()

    # check crds are installed
    ret = extensionv1.list_custom_resource_definition()
    found = 0
    for crd in ret.items:
        found = found + 1 if crd.metadata.name == 'aggregatorconfigs.diengine.opendilab.org' else found
        found = found + 1 if crd.metadata.name == 'dijobs.diengine.opendilab.org' else found
    assert found == 0

    # delete cluster
    launcher.delete_cluster()
    try:
        config.load_kube_config()
    except Exception:
        print("No k8s cluster found, skipped...")
    else:
        current_context = config.list_kube_config_contexts()[1]
        assert not current_context['context']['cluster'].startswith(f"k3d-{cluster_name}")
