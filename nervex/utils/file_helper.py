import logging
import os
import pickle
from typing import NoReturn, Union

import torch

from .import_utils import try_import_ceph

ceph = try_import_ceph()


def read_from_ceph(path: str) -> object:
    """
    Overview: read file from ceph
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
    Overview: read file from local file system
    Arguments:
        - path (:obj:`str`): file path in local file system
    Returns:
        - (:obj`data`): deserialized data
    """
    with open(path, "rb") as f:
        value = pickle.load(f)

    return value


def read_from_path(path: str):
    """
    Overview: read file from ceph
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
    Overview: save pickle dumped data file to ceph
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
    if fs_type is None:
        fs_type = 'ceph' if path.lower().startswith('s3') else 'normal'
    assert fs_type in ['normal', 'ceph']
    if fs_type == 'ceph':
        data = read_from_path(path)
    elif fs_type == 'normal':
        data = torch.load(path, map_location='cpu')
    return data


def save_file(path: str, data: object, fs_type: Union[None, str] = None) -> NoReturn:
    if fs_type is None:
        fs_type = 'ceph' if path.lower().startswith('s3') else 'normal'
    assert fs_type in ['normal', 'ceph']
    if fs_type == 'ceph':
        save_file_ceph(path, data)
    elif fs_type == 'normal':
        torch.save(data, path)


def remove_file(path: str, fs_type: Union[None, str] = None) -> NoReturn:
    if fs_type is None:
        fs_type = 'ceph' if path.lower().startswith('s3') else 'normal'
    assert fs_type in ['normal', 'ceph']
    if fs_type == 'ceph':
        pass
        os.popen("aws s3 rm --recursive {}".format(path))
    elif fs_type == 'normal':
        os.popen("rm -rf {}".format(path))
