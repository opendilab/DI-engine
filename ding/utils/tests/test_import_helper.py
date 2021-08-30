import pytest

from ding.utils.import_helper import try_import_ceph, try_import_mc, try_import_redis, try_import_rediscluster, \
    try_import_link, import_module, try_import_pickle, try_import_pyarrow


@pytest.mark.unittest
def test_try_import():
    try_import_ceph()
    try_import_mc()
    try_import_redis()
    try_import_rediscluster()
    try_import_link()
    import_module(['ding.utils'])
    try_import_pyarrow()
    try_import_pickle()
