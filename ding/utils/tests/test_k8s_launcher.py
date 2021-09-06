import pytest
import os
import subprocess
from ding.utils import K8sLauncher


@pytest.mark.unittest
def test_create_and_delete_k8s_cluster():
    cluster_name = 'test-k8s-launcher'
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'k8s-config.yaml')
    launcher = K8sLauncher(config_path)
    launcher.name = cluster_name
    launcher.create_cluster()
    proc = subprocess.Popen(['kubectl','config', 'current-context'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    assert err is not None
    assert out.decode('utf-8').startswith(f"k3d-{cluster_name}") 
    
    launcher.delete_cluster()
    proc = subprocess.Popen(['kubectl','config', 'current-context'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    assert err is not None
    assert not out.decode('utf-8').startswith(f"k3d-{cluster_name}") 