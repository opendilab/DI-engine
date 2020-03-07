import ceph
import numpy as np
from io import BytesIO


def read_file_ceph(path):
    s3client = ceph.S3Client()
    value = s3client.Get(path)
    if not value:
        raise FileNotFoundError("File({}) doesn't exist in ceph".format(path))
    value = BytesIO(value)
    return value
