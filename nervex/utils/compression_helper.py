from typing import Union, Callable, List, Optional
import copy
import pickle
import zlib

import lz4.block
import numpy as np
import torch


def dummy_compressor(data):
    return copy.deepcopy(data)


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
    return _COMPRESSORS_MAP[name]


def dummy_decompressor(data):
    return copy.deepcopy(data)


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
    return _DECOMPRESSORS_MAP[name]
