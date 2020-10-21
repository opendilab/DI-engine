import copy
import pickle
import zlib
from typing import Union, Callable, List

import lz4.block
import numpy as np
import torch


def function_listed(func: Callable[[
        object,
], object]) -> Callable[[
        object,
], Union[List[object], object]]:

    def _new_func(item: object) -> Union[List[object], object]:
        if isinstance(item, list):
            return [func(item) for item in item]
        else:
            return func(item)

    return _new_func


def compress_obs(obs):
    if obs is None:
        return None
    new_obs = {}
    special_list = ['entity_info', 'spatial_info']
    for k in obs.keys():
        if k not in special_list:
            new_obs[k] = obs[k]

    new_obs['entity_info'] = {}
    entity_no_bool = 4
    new_obs['entity_info'] = {}
    new_obs['entity_info']['no_bool'] = obs['entity_info'][:, :entity_no_bool].numpy()
    entity_bool = obs['entity_info'][:, entity_no_bool:].to(torch.uint8).numpy()
    new_obs['entity_info']['bool_ori_shape'] = entity_bool.shape
    B, N = entity_bool.shape
    N_strided = N if N % 8 == 0 else (N // 8 + 1) * 8
    new_obs['entity_info']['bool_strided_shape'] = (B, N_strided)
    if N != N_strided:
        entity_bool = np.concatenate([entity_bool, np.zeros((B, N_strided - N), dtype=np.uint8)], axis=1)
    new_obs['entity_info']['bool'] = np.packbits(entity_bool)

    spatial_no_bool = 1
    new_obs['spatial_info'] = {}
    spatial_bool = obs['spatial_info'][spatial_no_bool:].to(torch.uint8).numpy()
    spatial_uint8 = obs['spatial_info'][:spatial_no_bool].mul_(256).to(torch.uint8).numpy()
    new_obs['spatial_info']['no_bool'] = spatial_uint8
    new_obs['spatial_info']['bool_ori_shape'] = spatial_bool.shape
    new_obs['spatial_info']['bool'] = np.packbits(spatial_bool)
    return copy.deepcopy(new_obs)


def decompress_obs(obs):
    if obs is None:
        return None
    new_obs = {}
    special_list = ['entity_info', 'spatial_info']
    for k in obs.keys():
        if k not in special_list:
            new_obs[k] = obs[k]

    new_obs['entity_info'] = {}
    entity_bool = np.unpackbits(obs['entity_info']['bool']).reshape(*obs['entity_info']['bool_strided_shape'])
    if obs['entity_info']['bool_strided_shape'][1] != obs['entity_info']['bool_ori_shape'][1]:
        entity_bool = entity_bool[:, :obs['entity_info']['bool_ori_shape'][1]]
    entity_no_bool = obs['entity_info']['no_bool']
    spatial_bool = np.unpackbits(obs['spatial_info']['bool']).reshape(*obs['spatial_info']['bool_ori_shape'])
    spatial_uint8 = obs['spatial_info']['no_bool'].astype(np.float32) / 256.
    new_obs['entity_info'] = torch.cat([torch.FloatTensor(entity_no_bool), torch.FloatTensor(entity_bool)], dim=1)
    new_obs['spatial_info'] = torch.cat([torch.FloatTensor(spatial_uint8), torch.FloatTensor(spatial_bool)], dim=0)
    return copy.deepcopy(new_obs)


def dummy_compressor(step_data):
    return copy.deepcopy(step_data)


def simple_step_data_compressor(step_data):
    return {k: compress_obs(v) for k, v in step_data.items()}


def zlib_step_data_compressor(step_data):
    return zlib.compress(pickle.dumps({k: compress_obs(v) for k, v in step_data.items()}))


def lz4_step_data_compressor(step_data):
    return lz4.block.compress(pickle.dumps({k: compress_obs(v) for k, v in step_data.items()}))


_COMPRESSORS_MAP = {
    'simple': simple_step_data_compressor,
    'lz4': lz4_step_data_compressor,
    'zlib': zlib_step_data_compressor,
    'none': dummy_compressor,
}


def get_step_data_compressor(name: str):
    return function_listed(_COMPRESSORS_MAP[name])


def dummy_decompressor(step_data):
    return copy.deepcopy(step_data)


def simple_step_data_decompressor(compressed_step_data):
    return {k: decompress_obs(v) for k, v in compressed_step_data.items()}


def lz4_step_data_decompressor(compressed_step_data):
    return {k: decompress_obs(v) for k, v in pickle.loads(lz4.block.decompress(compressed_step_data)).items()}


def zlib_step_data_decompressor(compressed_step_data):
    return {k: decompress_obs(v) for k, v in pickle.loads(zlib.decompress(compressed_step_data)).items()}


_DECOMPRESSORS_MAP = {
    'simple': simple_step_data_decompressor,
    'lz4': lz4_step_data_decompressor,
    'zlib': zlib_step_data_decompressor,
    'none': dummy_decompressor,
}


def get_step_data_decompressor(name: str):
    return function_listed(_DECOMPRESSORS_MAP[name])
