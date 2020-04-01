from io import BytesIO
import pickle

from .import_utils import try_import_ceph

ceph = try_import_ceph()


def read_file_ceph(path, read_type='BytesIO'):
    """
    Overview: read file from ceph
    Arguments:
        - path (:obj:`str`): file path in ceph, start with "s3://"
        - read_type (:obj:`str`): choose in ['BytesIO', 'pickle']
    Returns:
        - (:obj`data`): fileStream or data
    """
    if ceph is None:
        import logging
        logging.warning(
            "You do not have ceph installed! Loading local file!"
            " If you are not testing locally, something is wrong!"
        )
        return open(path, "rb")

    s3client = ceph.S3Client()
    value = s3client.Get(path)
    if not value:
        raise FileNotFoundError("File({}) doesn't exist in ceph".format(path))
    if read_type == 'BytesIO':
        value = BytesIO(value)
    elif read_type == 'pickle':
        value = pickle.loads(value)
    return value


def save_file_ceph(save_path, file_name, data):
    """
    Overview: save file to ceph
    Arguments:
        - save_path (:obj:`str`): save root path in ceph, start with "s3://"
        - file_name (:obj:`str`): save file name in ceph
        - data (:obj:`angthing`): could be dict, list or tensor etc.
    """
    data = pickle.dumps(data)
    if ceph is not None:
        s3client = ceph.S3Client()
        s3client.save_from_string(save_path, file_name, data)
    else:
        import logging
        import os
        logging.warning(
            "You do not have ceph installed! Saving as local file!"
            " If you are not testing locally, something is wrong!"
        )
        with open(os.path.join(save_path, file_name), 'wb') as f:
            f.write(data)
