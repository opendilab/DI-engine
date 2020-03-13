from io import BytesIO

from .import_utils import try_import_ceph

ceph = try_import_ceph()


def read_file_ceph(path):

    if ceph is None:
        import logging
        logging.warning("You do not have ceph installed! If you are not testing locally, something is wrong!")
        return open(path, "rb")

    s3client = ceph.S3Client()
    value = s3client.Get(path)
    if not value:
        raise FileNotFoundError("File({}) doesn't exist in ceph".format(path))
    value = BytesIO(value)
    return value
