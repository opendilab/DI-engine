import torch
import torch.multiprocessing as mp
from array import array
import numpy as np


def shm_encode_with_schema(data, maxlen=1024 * 1024 * 64):
    idx = 0
    encoding = np.zeros(shape=(maxlen, ), dtype=np.float32)

    def _helper(data):
        nonlocal idx
        if isinstance(data, torch.Tensor):
            if data.size():
                tmp = data.flatten()
                encoding[idx:idx + len(tmp)] = tmp
                idx += len(tmp)
            else:
                encoding[idx] = data.item()
                idx += 1
            return data.size()
        elif isinstance(data, dict):
            schema = {}
            for key, chunk in data.items():
                schema[key] = _helper(chunk)
            return schema
        elif isinstance(data, (list, tuple)):
            schema = []
            for chunk in data:
                schema.append(_helper(chunk))
            if isinstance(data, tuple):
                schema = tuple(schema)
            return schema
        else:
            raise ValueError("not supported dtype! {}".format(type(data)))

    schema = _helper(data)
    return encoding[:idx], schema


def shm_decode(encoding, schema):

    def _shm_decode_helper(encoding, schema, start):
        if isinstance(schema, torch.Size):
            storage = 1
            for d in schema:
                storage *= d
            # if isinstance(encoding[start], int):
            #     decoding = torch.LongTensor(encoding[start: start + storage]).view(schema)
            # else:
            #     decoding = torch.FloatTensor(encoding[start: start + storage]).view(schema)
            decoding = torch.Tensor(encoding[start:start + storage]).view(schema)
            start += storage
            return decoding, start
        elif isinstance(schema, dict):
            decoding = {}
            for key, chunk in schema.items():
                decoding[key], start = _shm_decode_helper(encoding, chunk, start)
            return decoding, start
        elif isinstance(schema, (list, tuple)):
            decoding = []
            for chunk in schema:
                chunk, start = _shm_decode_helper(encoding, chunk, start)
                decoding.append(chunk)
            if isinstance(schema, tuple):
                decoding = tuple(decoding)
            return decoding, start
        else:
            raise ValueError("not supported schema! {}".format(schema))

    decoding, start = _shm_decode_helper(encoding, schema, 0)
    assert len(encoding) == start, "Encoding length and schema do not match!"
    return decoding


def equal(data1, data2, check_dict_order=True, strict_dtype=False, parent_key=None):
    if type(data1) != type(data2):
        print(parent_key)
        # print("data1: {} {}".format(parent_key, data1))
        # print("data2: {} {}".format(parent_key, data2))
        return False
    if isinstance(data1, torch.Tensor):
        if not strict_dtype:
            data1 = data1.float()
            data2 = data2.float()
        if data1.dtype != data2.dtype:
            print("data type does not match! data1({}), data2({})".format(data1.dtype, data2.dtype))
            # print("data1: {} {}".format(parent_key, data1))
            # print("data2: {} {}".format(parent_key, data2))
            return False
        if data1.equal(data2):
            return True
        else:
            print("value not match")
            # print("parent key of data1: {}".format(parent_key))
            print("data1: {} {}".format(parent_key, data1))
            print("data2: {} {}".format(parent_key, data2))
            return False
    elif isinstance(data1, dict):
        key_set1 = data1.keys()
        key_set2 = data2.keys()
        if check_dict_order and key_set1 != key_set2:
            print("key sequence not match!")
            # print("data1: {}".format(data1))
            # print("data2: {}".format(data2))
            return False
        elif set(key_set1) != set(key_set2):
            print("key set not match!")
            # print("data1: {}".format(data1))
            # print("data2: {}".format(data2))
            return False
        for key in key_set1:
            if equal(data1[key], data2[key], check_dict_order, strict_dtype, key):
                print("passed:", key)
            else:
                print("!!!!!!! not match:", key)
                return False
        return True
    elif isinstance(data1, (tuple, list)):
        if len(data1) != len(data2):
            print("list length does not match!")
            # print("data1: {}".format(data1))
            # print("data2: {}".format(data2))
            return False
        return all([equal(data1[i], data2[i], check_dict_order, strict_dtype, parent_key) for i in range(len(data1))])
    else:
        raise ValueError("not supported dtype! {}".format(data1))


def equal_v0(data1, data2, check_dict_order=True, strict_dtype=False):
    if type(data1) != type(data2):
        return False
    if isinstance(data1, torch.Tensor):
        if not strict_dtype:
            data1 = data1.float()
            data2 = data2.float()
        if data1.dtype != data2.dtype:
            print("data type does not match! data1({}), data2({})".format(data1.dtype, data2.dtype))
            return False
        return data1.equal(data2)
    elif isinstance(data1, dict):
        key_set1 = data1.keys()
        key_set2 = data2.keys()
        if check_dict_order and key_set1 != key_set2:
            return False
        elif set(key_set1) != set(key_set2):
            return False
        return all([equal(data1[key], data2[key], check_dict_order, strict_dtype) for key in key_set1])
    elif isinstance(data1, (tuple, list)):
        if len(data1) != len(data2):
            return False
        return all([equal(data1[i], data2[i], check_dict_order, strict_dtype) for i in range(len(data1))])
    else:
        raise ValueError("not supported dtype! {}".format(data1))