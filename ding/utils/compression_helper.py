import pickle
import zlib
import lz4.block
import numpy as np


def dummy_compressor(data):
    r"""
    Overview:
        Return input data.
    """
    return data


def zlib_data_compressor(data):
    r"""
    Overview:
        Takes the input compressed data and return the compressed original data (zlib compressor) in binary format.
    Examples:
        >>> zlib_data_compressor("Hello")
        b'x\x9ck`\x99\xca\xc9\x00\x01=\xac\x1e\xa999\xf9S\xf4\x00%L\x04j'
    """
    return zlib.compress(pickle.dumps(data))


def lz4_data_compressor(data):
    r"""
    Overview:
        Return the compressed original data (lz4 compressor).The compressor outputs in binary format.
    Examples:
        >>> lz4.block.compress(pickle.dumps("Hello"))
        b'\x14\x00\x00\x00R\x80\x04\x95\t\x00\x01\x00\x90\x8c\x05Hello\x94.'
    """
    return lz4.block.compress(pickle.dumps(data))


def jpeg_data_compressor(data):
    """
    Overview:
        To reduce memory usage, we can choose to store the jpeg strings of image
        instead of the numpy array in the buffer.
        This function encodes the observation numpy arr to the jpeg strings.
    Arguments:
        data (:obj:`np.array`): the observation numpy arr.
    """
    try:
        import cv2
    except ImportError:
        from ditk import logging
        import sys
        logging.warning("Please install opencv-python first.")
        sys.exit(1)
    img_str = cv2.imencode('.jpg', data)[1].tobytes()

    return img_str


_COMPRESSORS_MAP = {
    'lz4': lz4_data_compressor,
    'zlib': zlib_data_compressor,
    'jpeg': jpeg_data_compressor,
    'none': dummy_compressor,
}


def get_data_compressor(name: str):
    r"""
    Overview:
        Get the data compressor according to the input name
    Arguments:
        - name(:obj:`str`): Name of the compressor, support ``['lz4', 'zlib', 'jpeg', 'none']``
    Return:
        - (:obj:`Callable`): Corresponding data_compressor, taking input data returning compressed data.
    Example:
        >>> compress_fn = get_data_compressor('lz4')
        >>> compressed_data = compressed(input_data)
    """
    return _COMPRESSORS_MAP[name]


def dummy_decompressor(data):
    """
    Overview:
        Return input data.
    """
    return data


def lz4_data_decompressor(compressed_data):
    r"""
    Overview:
        Return the decompressed original data (lz4 compressor).
    """
    return pickle.loads(lz4.block.decompress(compressed_data))


def zlib_data_decompressor(compressed_data):
    r"""
    Overview:
        Return the decompressed original data (zlib compressor).
    """
    return pickle.loads(zlib.decompress(compressed_data))


def jpeg_data_decompressor(compressed_data, gray_scale=False):
    """
    Overview:
        To reduce memory usage, we can choose to store the jpeg strings of image
        instead of the numpy array in the buffer.
        This function decodes the observation numpy arr from the jpeg strings.
    Arguments:
        compressed_data (:obj:`str`): the jpeg strings.
        gray_scale (:obj:`bool`): if the observation is gray, ``gray_scale=True``,
            if the observation is RGB, ``gray_scale=False``.
    """
    try:
        import cv2
    except ImportError:
        from ditk import logging
        import sys
        logging.warning("Please install opencv-python first.")
        sys.exit(1)
    nparr = np.frombuffer(compressed_data, np.uint8)
    if gray_scale:
        arr = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        arr = np.expand_dims(arr, -1)
    else:
        arr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return arr


_DECOMPRESSORS_MAP = {
    'lz4': lz4_data_decompressor,
    'zlib': zlib_data_decompressor,
    'jpeg': jpeg_data_decompressor,
    'none': dummy_decompressor,
}


def get_data_decompressor(name: str):
    r"""
    Overview:
        Get the data decompressor according to the input name
    Arguments:
        - name(:obj:`str`): Name of the decompressor, support ``['lz4', 'zlib', 'none']``

    .. note::

        For all the decompressors, the input of a bytes-like object is required.

    Returns:
        - (:obj:`Callable`): Corresponding data_decompressor.
    Examples:
        >>> decompress_fn = get_data_decompressor('lz4')
        >>> origin_data = compressed(compressed_data)
    """
    return _DECOMPRESSORS_MAP[name]
