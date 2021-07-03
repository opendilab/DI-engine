import pytest

from nervex.utils.import_helper import try_import_ceph, try_import_mc, try_import_redis, try_import_rediscluster, \
    try_import_link, import_module


@pytest.mark.unittest
def test_try_import():
    try_import_ceph()
    try_import_mc()
    try_import_redis()
    try_import_rediscluster()
    try_import_link()
    import_module(['nervex.utils'])
