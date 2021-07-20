import importlib
from typing import List

import ding
from .default_helper import one_time_warning


def try_import_ceph():
    """
    Overview:
        Try import ceph module, if failed, return ``None``

    Returns:
        - (:obj:`Module`): Imported module, or ``None`` when ceph not found
    """
    try:
        import ceph
        client = ceph.S3Client()
        return client
    except ModuleNotFoundError as e:
        try:
            from petrel_client.client import Client
            client = Client(conf_path='~/petreloss.conf')
            return client
        except ModuleNotFoundError as e:
            one_time_warning("You have not installed ceph package! DI-engine has changed to some alternatives.")
            ceph = None
            return ceph


def try_import_mc():
    """
    Overview:
        Try import mc module, if failed, return ``None``

    Returns:
        - (:obj:`Module`): Imported module, or ``None`` when mc not found
    """
    try:
        import mc
    except ModuleNotFoundError as e:
        one_time_warning("You have not installed memcache package! DI-engine has changed to some alternatives.")
        mc = None
    return mc


def try_import_redis():
    """
    Overview:
        Try import redis module, if failed, return ``None``

    Returns:
        - (:obj:`Module`): Imported module, or ``None`` when redis not found
    """
    try:
        import redis
    except ModuleNotFoundError as e:
        one_time_warning("You have not installed redis package! DI-engine has changed to some alternatives.")
        redis = None
    return redis


def try_import_rediscluster():
    """
    Overview:
        Try import rediscluster module, if failed, return ``None``

    Returns:
        - (:obj:`Module`): Imported module, or ``None`` when rediscluster not found
    """
    try:
        import rediscluster
    except ModuleNotFoundError as e:
        one_time_warning("You have not installed rediscluster package! DI-engine has changed to some alternatives.")
        rediscluster = None
    return rediscluster


def try_import_link():
    """
    Overview:
        Try import linklink module, if failed, import ding.tests.fake_linklink instead

    Returns:
        - (:obj:`Module`): Imported module (may be ``fake_linklink``)
    """
    if ding.enable_linklink:
        try:
            import linklink as link
        except ModuleNotFoundError as e:
            one_time_warning("You have not installed linklink package! DI-engine has changed to some alternatives.")
            from .fake_linklink import link
    else:
        from .fake_linklink import link

    return link


def import_module(modules: List[str]) -> None:
    """
    Overview:
        Import several module as a list
    Arguments:
        - (:obj:`str list`): List of module names
    """
    for name in modules:
        importlib.import_module(name)
