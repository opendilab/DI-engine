import logging
import os
import pickle
from typing import NoReturn, Union

import torch
from pathlib import Path
import io
import time

from .import_helper import try_import_ceph, try_import_redis, try_import_rediscluster, try_import_mc
from .lock_helper import get_rw_lock

global r, rc, mclient
mclient = None
r = None
rc = None

ceph = try_import_ceph()
# mc = try_import_mc()
mc = None
redis = try_import_redis()
rediscluster = try_import_rediscluster()


def read_from_ceph(path: str) -> object:
    """
    Overview:
        Read file from ceph
    Arguments:
        - path (:obj:`str`): File path in ceph, start with ``"s3://"``
    Returns:
        - (:obj:`data`): Deserialized data
    """
    value = ceph.Get(path)
    if not value:
        raise FileNotFoundError("File({}) doesn't exist in ceph".format(path))

    return pickle.loads(value)


def _ensure_redis(host='localhost', port=6379):
    """
    Overview: Ensures redis usage
    Arguments:
        - host (:obj:`str`): Host string
        - port (:obj:`int`): Port number
    Returns:
        - (:obj:`Redis(object)`): Redis object with given ``host``, ``port``, and ``db=0``
    """
    global r
    if r is None:
        r = redis.StrictRedis(host=host, port=port, db=0)
    return


def read_from_redis(path: str) -> object:
    """
    Overview:
        Read file from redis
    Arguments:
        - path (:obj:`str`): Dile path in redis, could be a string key
    Returns:
        - (:obj:`data`): Deserialized data
    """
    global r
    _ensure_redis()
    value_bytes = r.get(path)
    value = pickle.loads(value_bytes)
    return value


def _ensure_rediscluster(startup_nodes=[{"host": "127.0.0.1", "port": "7000"}]):
    """
    Overview:
        Ensures redis usage
    Arguments:
        - List of startup nodes (:obj:`dict`) of
            - host (:obj:`str`): Host string
            - port (:obj:`int`): Port number
    Returns:
        - (:obj:`RedisCluster(object)`): RedisCluster object with given ``host``, ``port``, \
            and ``False`` for ``decode_responses`` in default.
    """
    global rc
    if rc is None:
        rc = rediscluster.RedisCluster(startup_nodes=startup_nodes, decode_responses=False)
    return


def read_from_rediscluster(path: str) -> object:
    """
    Overview:
        Read file from rediscluster
    Arguments:
        - path (:obj:`str`): Dile path in rediscluster, could be a string key
    Returns:
        - (:obj:`data`): Deserialized data
    """
    global rc
    _ensure_rediscluster()
    value_bytes = rc.get(path)
    value = pickle.loads(value_bytes)
    return value


def read_from_file(path: str) -> object:
    """
    Overview:
        Read file from local file system
    Arguments:
        - path (:obj:`str`): File path in local file system
    Returns:
        - (:obj:`data`): Deserialized data
    """
    with open(path, "rb") as f:
        value = pickle.load(f)

    return value


def _ensure_memcached():
    """
    Overview:
        Ensures memcache usage
    Returns:
        - (:obj:`MemcachedClient instance`): MemcachedClient's class instance built with current \
            memcached_client's ``server_list.conf`` and ``client.conf`` files
    """
    global mclient
    if mclient is None:
        server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
        client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
        mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
    return


def read_from_mc(path: str, flush=False) -> object:
    """
    Overview:
        Read file from memcache, file must be saved by `torch.save()`
    Arguments:
        - path (:obj:`str`): File path in local system
    Returns:
        - (:obj:`data`): Deserialized data
    """
    global mclient
    _ensure_memcached()
    while True:
        try:
            value = mc.pyvector()
            if flush:
                mclient.Get(path, value, mc.MC_READ_THROUGH)
                return
            else:
                mclient.Get(path, value)
            value_buf = mc.ConvertBuffer(value)
            value_str = io.BytesIO(value_buf)
            value_str = torch.load(value_str, map_location='cpu')
            return value_str
        except:
            print('read mc failed, retry...')
            time.sleep(0.01)


def read_from_path(path: str):
    """
    Overview:
        Read file from ceph
    Arguments:
        - path (:obj:`str`): File path in ceph, start with ``"s3://"``, or use local file system
    Returns:
        - (:obj:`data`): Deserialized data
    """
    if ceph is None:
        logging.info(
            "You do not have ceph installed! Loading local file!"
            " If you are not testing locally, something is wrong!"
        )
        return read_from_file(path)
    else:
        return read_from_ceph(path)


def save_file_ceph(path, data):
    """
    Overview:
        Save pickle dumped data file to ceph
    Arguments:
        - path (:obj:`str`): File path in ceph, start with ``"s3://"``, use file system when not
        - data (:obj:`Any`): Could be dict, list or tensor etc.
    """
    data = pickle.dumps(data)
    save_path = os.path.dirname(path)
    file_name = os.path.basename(path)
    if ceph is not None:
        if hasattr(ceph, 'save_from_string'):
            ceph.save_from_string(save_path, file_name, data)
        elif hasattr(ceph, 'put'):
            ceph.put(os.path.join(save_path, file_name), data)
        else:
            raise RuntimeError('ceph can not save file, check your ceph installation')
    else:
        import logging
        size = len(data)
        if save_path == 'do_not_save':
            logging.info(
                "You do not have ceph installed! ignored file {} of size {}!".format(file_name, size) +
                " If you are not testing locally, something is wrong!"
            )
            return
        p = os.path.join(save_path, file_name)
        with open(p, 'wb') as f:
            logging.info(
                "You do not have ceph installed! Saving as local file at {} of size {}!".format(p, size) +
                " If you are not testing locally, something is wrong!"
            )
            f.write(data)


def save_file_redis(path, data):
    """
    Overview:
        Save pickle dumped data file to redis
    Arguments:
        - path (:obj:`str`): File path (could be a string key) in redis
        - data (:obj:`Any`): Could be dict, list or tensor etc.
    """
    global r
    _ensure_redis()
    data = pickle.dumps(data)
    r.set(path, data)
    return


def save_file_rediscluster(path, data):
    """
    Overview:
        Save pickle dumped data file to rediscluster
    Arguments:
        - path (:obj:`str`): File path (could be a string key) in redis
        - data (:obj:`Any`): Could be dict, list or tensor etc.
    """
    global rc
    _ensure_rediscluster()
    data = pickle.dumps(data)
    rc.set(path, data)
    return


def read_file(path: str, fs_type: Union[None, str] = None, use_lock: bool = False) -> object:
    r"""
    Overview:
        Read file from path
    Arguments:
        - path (:obj:`str`): The path of file to read
        - fs_type (:obj:`str` or :obj:`None`): The file system type, support ``{'normal', 'ceph'}``
        - use_lock (:obj:`bool`): Whether ``use_lock`` is in local normal file system
    """
    if fs_type is None:
        if path.lower().startswith('s3'):
            fs_type = 'ceph'
        elif mc is not None:
            fs_type = 'mc'
        else:
            fs_type = 'normal'
    assert fs_type in ['normal', 'ceph', 'mc']
    if fs_type == 'ceph':
        data = read_from_path(path)
    elif fs_type == 'normal':
        if use_lock:
            with get_rw_lock(path, 'read'):
                data = torch.load(path, map_location='cpu')
        else:
            data = torch.load(path, map_location='cpu')
    elif fs_type == 'mc':
        data = read_from_mc(path)
    return data


def save_file(path: str, data: object, fs_type: Union[None, str] = None, use_lock: bool = False) -> NoReturn:
    r"""
    Overview:
        Save data to file of path
    Arguments:
        - path (:obj:`str`): The path of file to save to
        - data (:obj:`object`): The data to save
        - fs_type (:obj:`str` or :obj:`None`): The file system type, support ``{'normal', 'ceph'}``
        - use_lock (:obj:`bool`): Whether ``use_lock`` is in local normal file system
    """
    if fs_type is None:
        if path.lower().startswith('s3'):
            fs_type = 'ceph'
        elif mc is not None:
            fs_type = 'mc'
        else:
            fs_type = 'normal'
    assert fs_type in ['normal', 'ceph', 'mc']
    if fs_type == 'ceph':
        save_file_ceph(path, data)
    elif fs_type == 'normal':
        if use_lock:
            with get_rw_lock(path, 'write'):
                torch.save(data, path)
        else:
            torch.save(data, path)
    elif fs_type == 'mc':
        torch.save(data, path)
        read_from_mc(path, flush=True)


def remove_file(path: str, fs_type: Union[None, str] = None) -> NoReturn:
    r"""
    Overview:
        Remove file
    Arguments:
        - path (:obj:`str`): The path of file you want to remove
        - fs_type (:obj:`str` or :obj:`None`): The file system type, support ``{'normal', 'ceph'}``
    """
    if fs_type is None:
        fs_type = 'ceph' if path.lower().startswith('s3') else 'normal'
    assert fs_type in ['normal', 'ceph']
    if fs_type == 'ceph':
        pass
        os.popen("aws s3 rm --recursive {}".format(path))
    elif fs_type == 'normal':
        os.popen("rm -rf {}".format(path))
