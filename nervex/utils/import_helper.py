import importlib
import logging
from typing import List
import nervex

global ceph_flag, redis_flag, rediscluster_flag, linklink_flag, mc_flag
ceph_flag, redis_flag, rediscluster_flag, linklink_flag, mc_flag = True, True, True, True, True


def try_import_ceph():
    """
    Overview:
        Try import ceph module, if failed, return None

    Returns:
        module: imported module, or None when ceph not found
    """
    global ceph_flag
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
            if ceph_flag:
                logging.warning("You have not installed ceph package! nervex has changed to some alternatives.")
            ceph = None
            ceph_flag = False
            return ceph


def try_import_mc():
    """
    Overview:
        Try import mc module, if failed, return None

    Returns:
        module: imported module, or None when mc not found
    """
    global mc_flag
    try:
        import mc
    except ModuleNotFoundError as e:
        if mc_flag:
            logging.warning("You have not installed memcache package! nervex has changed to some alternatives.")
        mc = None
        mc_flag = False
    return mc


def try_import_redis():
    """
    Overview:
        Try import redis module, if failed, return None

    Returns:
        module: imported module, or None when redis not found
    """
    global redis_flag
    try:
        import redis
    except ModuleNotFoundError as e:
        if redis_flag:
            logging.warning("You have not installed redis package! nervex has changed to some alternatives.")
        redis = None
        redis_flag = False
    return redis


def try_import_rediscluster():
    """
    Overview:
        Try import rediscluster module, if failed, return None

    Returns:
        module: imported module, or None when rediscluster not found
    """
    global rediscluster_flag
    try:
        import rediscluster
    except ModuleNotFoundError as e:
        if rediscluster_flag:
            logging.warning("You have not installed rediscluster package! nervex has changed to some alternatives.")
        rediscluster = None
        rediscluster_flag = False
    return rediscluster


def try_import_link():
    global linklink_flag
    """
    Overview:
        Try import linklink module, if failed, import nervex.tests.fake_linklink instead

    Returns:
        module: imported module (may be fake_linklink)
    """
    if nervex.enable_linklink:
        try:
            import linklink as link
        except ModuleNotFoundError as e:
            if linklink_flag:
                logging.warning("You have not installed linklink package! nervex has changed to some alternatives.")
            from .fake_linklink import link
            linklink_flag = False
    else:
        from .fake_linklink import link
        linklink_flag = False
        
    return link


def import_module(modules: List[str]) -> None:
    """
    Overview:
        Import several module as a list
    Args:
        - modules (:obj:`list` of `str`): List of module names
    """
    for name in modules:
        importlib.import_module(name)
