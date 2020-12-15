import logging
import os
import pickle
from typing import NoReturn, Union

import torch
from pathlib import Path
import io

from .import_helper import try_import_ceph
from .import_helper import try_import_mc

global mclient
mclient = None

ceph = try_import_ceph()
mc = try_import_mc()


def read_from_ceph(path: str) -> object:
    """
    Overview:
        read file from ceph
    Arguments:
        - path (:obj:`str`): file path in ceph, start with "s3://"
    Returns:
        - (:obj`data`): deserialized data
    """
    s3client = ceph.S3Client()
    value = s3client.Get(path)
    if not value:
        raise FileNotFoundError("File({}) doesn't exist in ceph".format(path))

    return pickle.loads(value)


def read_from_file(path: str) -> object:
    """
    Overview:
        read file from local file system
    Arguments:
        - path (:obj:`str`): file path in local file system
    Returns:
        - (:obj`data`): deserialized data
    """
    with open(path, "rb") as f:
        value = pickle.load(f)

    return value


def _ensure_memcached():
    global mclient
    if mclient is None:
        server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
        client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
        mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
    return


def read_from_mc(path: str, flush=False) -> object:
    """
    Overview:
        read file from memcache, file must be saved by `torch.save()`
    Arguments:
        - path (:obj:`str`): file path in local system
    Returns:
        - (:obj`data`): deserialized data
    """
    global mclient
    _ensure_memcached()
    value = mc.pyvector()
    if flush:
        mclient.Get(path, value, mc.MC_READ_THROUGH)
    else:
        mclient.Get(path, value)
    value_buf = mc.ConvertBuffer(value)
    value_str = io.BytesIO(value_buf)
    value_str = torch.load(value_str)

    return value_str


def read_from_path(path: str):
    """
    Overview:
        read file from ceph
    Arguments:
        - path (:obj:`str`): file path in ceph, start with "s3://", or use local file system
    Returns:
        - (:obj`data`): deserialized data
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
        save pickle dumped data file to ceph
    Arguments:
        - path (:obj:`str`): file path in ceph, start with "s3://", use file system when not
        - data (:obj:`anything`): could be dict, list or tensor etc.
    """
    data = pickle.dumps(data)
    save_path = os.path.dirname(path)
    file_name = os.path.basename(path)
    if ceph is not None:
        s3client = ceph.S3Client()
        s3client.save_from_string(save_path, file_name, data)
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


def read_file(path: str, fs_type: Union[None, str] = None) -> object:
    r"""
    Overview:
        read file from path
    Arguments:
        - path (:obj:`str`): the path of file to read
        - fs_type (:obj:`str` or :obj:`None`): the file system type, support 'normal' and 'ceph'
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
        data = torch.load(path, map_location='cpu')
    elif fs_type == 'mc':
        data = read_from_mc(path)
    return data


def save_file(path: str, data: object, fs_type: Union[None, str] = None) -> NoReturn:
    r"""
    Overview:
        save data to file of path
    Arguments:
        - path (:obj:`str`): the path of file to save to
        - data (:obj:`object`): the data to save
        - fs_type (:obj:`str` or :obj:`None`): the file system type, support 'normal' and 'ceph'
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
        torch.save(data, path)
    elif fs_type == 'mc':
        torch.save(data, path)
        read_from_mc(path, flush=True)


def remove_file(path: str, fs_type: Union[None, str] = None) -> NoReturn:
    r"""
    Overview:
        remove file
    Arguments:
        - path (:obj:`str`): the path of file you want to remove
        - fs_type (:obj:`str` or :obj:`None`): the file system type, support 'normal' and 'ceph'
    """
    if fs_type is None:
        fs_type = 'ceph' if path.lower().startswith('s3') else 'normal'
    assert fs_type in ['normal', 'ceph']
    if fs_type == 'ceph':
        pass
        os.popen("aws s3 rm --recursive {}".format(path))
    elif fs_type == 'normal':
        os.popen("rm -rf {}".format(path))
