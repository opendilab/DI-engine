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
        logging.info(
            "You do not have ceph installed! Loading local file!"
            " If you are not testing locally, something is wrong!"
        )
        f = open(path, "rb")
        if read_type == 'BytesIO':
            return f
        elif read_type == 'pickle':
            value = pickle.load(f)
            f.close()
            return value
    else:
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
    Overview: save pickle dumped data file to ceph
    Arguments:
        - save_path (:obj:`str`): save root path in ceph, start with "s3://"
        - file_name (:obj:`str`): save file name in ceph
        - data (:obj:`anything`): could be dict, list or tensor etc.
    """
    data = pickle.dumps(data)
    if ceph is not None:
        s3client = ceph.S3Client()
        s3client.save_from_string(save_path, file_name, data)
    else:
        import logging
        import os
        size = len(data)
        if(save_path == 'do_not_save'):
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
