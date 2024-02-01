from typing import Any, ByteString, Callable
import pickle
import cloudpickle
import zlib
import numpy as np


class CloudPickleWrapper:
    """
    Overview:
        CloudPickleWrapper can be able to pickle more python object(e.g: an object with lambda expression).
    Interfaces:
        ``__init__``, ``__getstate__``, ``__setstate__``.
    """

    def __init__(self, data: Any) -> None:
        """
        Overview:
            Initialize the CloudPickleWrapper using the given arguments.
        Arguments:
            - data (:obj:`Any`): The object to be dumped.
        """
        self.data = data

    def __getstate__(self) -> bytes:
        """
        Overview:
            Get the state of the CloudPickleWrapper.
        Returns:
            - data (:obj:`bytes`): The dumped byte-like result.
        """

        return cloudpickle.dumps(self.data)

    def __setstate__(self, data: bytes) -> None:
        """
        Overview:
            Set the state of the CloudPickleWrapper.
        Arguments:
            - data (:obj:`bytes`): The dumped byte-like result.
        """

        if isinstance(data, (tuple, list, np.ndarray)):  # pickle is faster
            self.data = pickle.loads(data)
        else:
            self.data = cloudpickle.loads(data)


def dummy_compressor(data: Any) -> Any:
    """
    Overview:
        Return the raw input data.
    Arguments:
        - data (:obj:`Any`): The input data of the compressor.
    Returns:
        - output (:obj:`Any`): This compressor will exactly return the input data.
    """
    return data


def zlib_data_compressor(data: Any) -> bytes:
    """
    Overview:
        Takes the input compressed data and return the compressed original data (zlib compressor) in binary format.
    Arguments:
        - data (:obj:`Any`): The input data of the compressor.
    Returns:
        - output (:obj:`bytes`): The compressed byte-like result.
    Examples:
        >>> zlib_data_compressor("Hello")
    """
    return zlib.compress(pickle.dumps(data))


def lz4_data_compressor(data: Any) -> bytes:
    """
    Overview:
        Return the compressed original data (lz4 compressor).The compressor outputs in binary format.
    Arguments:
        - data (:obj:`Any`): The input data of the compressor.
    Returns:
        - output (:obj:`bytes`): The compressed byte-like result.
    Examples:
        >>> lz4.block.compress(pickle.dumps("Hello"))
        b'\x14\x00\x00\x00R\x80\x04\x95\t\x00\x01\x00\x90\x8c\x05Hello\x94.'
    """
    try:
        import lz4.block
    except ImportError:
        from ditk import logging
        import sys
        logging.warning("Please install lz4 first, such as `pip3 install lz4`")
        sys.exit(1)
    return lz4.block.compress(pickle.dumps(data))


def jpeg_data_compressor(data: np.ndarray) -> bytes:
    """
    Overview:
        To reduce memory usage, we can choose to store the jpeg strings of image instead of the numpy array in \
        the buffer. This function encodes the observation numpy arr to the jpeg strings.
    Arguments:
        - data (:obj:`np.array`): the observation numpy arr.
    Returns:
        - img_str (:obj:`bytes`): The compressed byte-like result.
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
    """
    Overview:
        Get the data compressor according to the input name.
    Arguments:
        - name(:obj:`str`): Name of the compressor, support ``['lz4', 'zlib', 'jpeg', 'none']``
    Return:
        - compressor (:obj:`Callable`): Corresponding data_compressor, taking input data returning compressed data.
    Example:
        >>> compress_fn = get_data_compressor('lz4')
        >>> compressed_data = compressed(input_data)
    """
    return _COMPRESSORS_MAP[name]


def dummy_decompressor(data: Any) -> Any:
    """
    Overview:
        Return the input data.
    Arguments:
        - data (:obj:`Any`): The input data of the decompressor.
    Returns:
        - output (:obj:`bytes`): The decompressed result, which is exactly the input.
    """
    return data


def lz4_data_decompressor(compressed_data: bytes) -> Any:
    """
    Overview:
        Return the decompressed original data (lz4 compressor).
    Arguments:
        - data (:obj:`bytes`): The input data of the decompressor.
    Returns:
        - output (:obj:`Any`): The decompressed object.
    """
    try:
        import lz4.block
    except ImportError:
        from ditk import logging
        import sys
        logging.warning("Please install lz4 first, such as `pip3 install lz4`")
        sys.exit(1)
    return pickle.loads(lz4.block.decompress(compressed_data))


def zlib_data_decompressor(compressed_data: bytes) -> Any:
    """
    Overview:
        Return the decompressed original data (zlib compressor).
    Arguments:
        - data (:obj:`bytes`): The input data of the decompressor.
    Returns:
        - output (:obj:`Any`): The decompressed object.
    """
    return pickle.loads(zlib.decompress(compressed_data))


def jpeg_data_decompressor(compressed_data: bytes, gray_scale=False) -> np.ndarray:
    """
    Overview:
        To reduce memory usage, we can choose to store the jpeg strings of image instead of the numpy array in the \
        buffer. This function decodes the observation numpy arr from the jpeg strings.
    Arguments:
        - compressed_data (:obj:`bytes`): The jpeg strings.
        - gray_scale (:obj:`bool`): If the observation is gray, ``gray_scale=True``,
            if the observation is RGB, ``gray_scale=False``.
    Returns:
        - arr (:obj:`np.ndarray`): The decompressed numpy array.
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


def get_data_decompressor(name: str) -> Callable:
    """
    Overview:
        Get the data decompressor according to the input name.
    Arguments:
        - name(:obj:`str`): Name of the decompressor, support ``['lz4', 'zlib', 'none']``

    .. note::

        For all the decompressors, the input of a bytes-like object is required.

    Returns:
        - decompressor (:obj:`Callable`): Corresponding data decompressor.
    Examples:
        >>> decompress_fn = get_data_decompressor('lz4')
        >>> origin_data = compressed(compressed_data)
    """
    return _DECOMPRESSORS_MAP[name]
