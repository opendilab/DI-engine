import os
import pickle

import torch

from .import_utils import try_import_ceph

ceph = try_import_ceph()


def read_file_ceph(path):
    """
    Overview: read file from ceph
    Arguments:
        - path (:obj:`str`): file path in ceph, start with "s3://"
    Returns:
        - (:obj`data`): fileStream or data
    """
    if ceph is None:
        import logging
        logging.info(
            "You do not have ceph installed! Loading local file!"
            " If you are not testing locally, something is wrong!"
        )
        with open(path, "rb") as f:
            value = pickle.load(f)
        return value
    else:
        s3client = ceph.S3Client()
        value = s3client.Get(path)
        if not value:
            raise FileNotFoundError("File({}) doesn't exist in ceph".format(path))
        value = pickle.loads(value)
        return value


def save_file_ceph(path, data):
    """
    Overview: save pickle dumped data file to ceph
    Arguments:
        - path (:obj:`str`): file path in ceph, start with "s3://"
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
        if (save_path == 'do_not_save'):
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


def read_file(path, fs_type=None):
    if fs_type is None:
        fs_type = 'ceph' if path.lower().startswith('s3') else 'normal'
    assert fs_type in ['normal', 'ceph']
    if fs_type == 'ceph':
        data = read_file_ceph(path)
    elif fs_type == 'normal':
        data = torch.load(path, map_location='cpu')
    return data


def save_file(path, data, fs_type=None):
    if fs_type is None:
        fs_type = 'ceph' if path.lower().startswith('s3') else 'normal'
    assert fs_type in ['normal', 'ceph']
    if fs_type == 'ceph':
        save_file_ceph(path, data)
    elif fs_type == 'normal':
        torch.save(data, path)
