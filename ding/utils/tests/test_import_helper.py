import pytest

import ding
from ding.utils.import_helper import try_import_ceph, try_import_mc, try_import_redis, try_import_rediscluster, \
    try_import_link, import_module


@pytest.mark.unittest
def test_try_import():
    try_import_ceph()
    try_import_mc()
    try_import_redis()
    try_import_rediscluster()
    try_import_link()
    import_module(['ding.utils'])
    ding.enable_linklink = True
    try_import_link()
