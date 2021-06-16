from typing import Union, Callable, List, Optional
import copy
import pickle
import zlib

import lz4.block
import numpy as np
import torch


def dummy_compressor(data):
    return data


def zlib_data_compressor(data):
    return zlib.compress(pickle.dumps(data))


def lz4_data_compressor(data):
    return lz4.block.compress(pickle.dumps(data))


_COMPRESSORS_MAP = {
    'lz4': lz4_data_compressor,
    'zlib': zlib_data_compressor,
    'none': dummy_compressor,
}


def get_data_compressor(name: str):
    r"""
    Overview:
        get the data compressor according to the input name
    Arguments:
        - name(:obj:`str`): the name of the compressor, support ['lz4', 'zlib', 'none']
    Return:
        - (:obj:`Callable`): the corresponding data_compressor funcation, \
            which will takes the input data and return the compressed data.
    Example:
        >>> compress_fn = get_data_compressor('lz4')
        >>> compressed_data = compressed(input_data)
    """
    return _COMPRESSORS_MAP[name]


def dummy_decompressor(data):
    return data


def lz4_data_decompressor(compressed_data):
    return pickle.loads(lz4.block.decompress(compressed_data))


def zlib_data_decompressor(compressed_data):
    return pickle.loads(zlib.decompress(compressed_data))


_DECOMPRESSORS_MAP = {
    'lz4': lz4_data_decompressor,
    'zlib': zlib_data_decompressor,
    'none': dummy_decompressor,
}


def get_data_decompressor(name: str):
    r"""
    Overview:
        get the data decompressor according to the input name
    Arguments:
        - name(:obj:`str`): the name of the decompressor, support ['lz4', 'zlib', 'none']
    Return:
        - (:obj:`Callable`): the corresponding data_decompressor funcation, \
            which will takes the input compressed data and return the decompressed original data.
    Example:
        >>> decompress_fn = get_data_decompressor('lz4')
        >>> origin_data = compressed(compressed_data)
    """
    return _DECOMPRESSORS_MAP[name]
