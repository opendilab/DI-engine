import copy
import json
import os
from typing import NoReturn, Optional, List

import yaml
from easydict import EasyDict


def read_config(path: str) -> EasyDict:
    """
    Args:
        path (str): Path of source yaml

    Returns:
        EasyDict: Config data from this file with dict type
    """
    with open(path, "r") as f:
        config_ = yaml.safe_load(f)

    return EasyDict(config_)


def save_config(config_: dict, path: str) -> NoReturn:
    """
    Args:
        config_ (dict): Config data
        path (str): Path of target yaml
    """
    config_string = json.dumps(config_)
    with open(path, "w") as f:
        yaml.safe_dump(json.loads(config_string), f)


def deep_merge_dicts(original: dict, new_dict: dict) -> dict:
    """
    Args:
        original (dict): Dict 1.
        new_dict (dict): Dict 2.

    Returns:
         dict: A new dict that is d1 and d2 deeply merged.
    """
    original = original or {}
    new_dict = new_dict or {}
    merged = copy.deepcopy(original)
    if new_dict:  # if new_dict is neither empty dict nor None
        deep_update(merged, new_dict, True, [])

    return merged


def deep_update(original: dict, new_dict: dict, new_keys_allowed: bool = False,
                whitelist: Optional[List[str]] = None,
                override_all_if_type_changes: Optional[List[str]] = None):
    """Updates original dict with values from new_dict recursively.

    pzh: It's only a function to merge new_dict into original. This is it.

    If new key is introduced in new_dict, then if new_keys_allowed is not
    True, an error will be thrown. Further, for sub-dicts, if the key is
    in the whitelist, then new subkeys can be introduced.

    Args:
        original (dict): Dictionary with default values.
        new_dict (dict): Dictionary with values to be updated
        new_keys_allowed (bool): Whether new keys are allowed.
        whitelist (Optional[List[str]]): List of keys that correspond to dict
            values where new subkeys can be introduced. This is only at the top
            level.
        override_all_if_type_changes(Optional[List[str]]): List of top level
            keys with value=dict, for which we always simply override the
            entire value (dict), if the "type" key in that value dict changes.
    """
    whitelist = whitelist or []
    override_all_if_type_changes = override_all_if_type_changes or []

    for k, value in new_dict.items():
        if k not in original and not new_keys_allowed:
            raise Exception("Unknown config parameter `{}`. Base config have: {}.".format(k, original.keys()))

        # Both original value and new one are dicts.
        if isinstance(original.get(k), dict) and isinstance(value, dict):
            # Check old type vs old one. If different, override entire value.
            if k in override_all_if_type_changes and \
                    "type" in value and "type" in original[k] and \
                    value["type"] != original[k]["type"]:
                original[k] = value
            # Whitelisted key -> ok to add new subkeys.
            elif k in whitelist:
                deep_update(original[k], value, True)
            # Non-whitelisted key.
            else:
                deep_update(original[k], value, new_keys_allowed)
        # Original value not a dict OR new value not a dict:
        # Override entire value.
        else:
            original[k] = value
    return original


def flatten_dict(data: dict, delimiter: str = "/") -> dict:
    """
        Flatten nested dict:
            {
                'a': {
                    'aa': {'aaa': data-aaa},
                    'ab': data-ab
                }
            }
        to:
            {
                'a/ab': data-ab,
                'a/aa/aaa': data-aaa
            }

    Args:
        data (dict): Original nested dict
        delimiter (str): Delimiter of the keys of the new dict

    Returns:
        dict: Flattened nested dict
    """
    data = copy.deepcopy(data)
    while any(isinstance(v, dict) for v in data.values()):
        remove = []
        add = {}
        for key, value in data.items():
            if isinstance(value, dict):
                for subkey, v in value.items():
                    add[delimiter.join([key, subkey])] = v
                remove.append(key)
        data.update(add)
        for k in remove:
            del data[k]
    return data


if __name__ == '__main__':
    import tempfile

    # Test whether save and read is reversible.
    old_config = EasyDict(
        {
            "aa": 1,
            "bb": 0.0001,
            "cc": None,
            "dd": "string",
            "ee": ["11", "22"],
            "ff": {
                "correct": 11
            }
        }
    )
    yaml_path = tempfile.mktemp(suffix=".yaml")
    save_config(old_config, yaml_path)
    assert os.path.exists(yaml_path)
    config = read_config(yaml_path)


    def assert_equal(item1, iterm2):
        if isinstance(item1, list):
            for item11, iterm22 in zip(item1, iterm2):
                assert_equal(item11, iterm22)
        elif isinstance(item1, dict):
            for item11, item22 in zip(item1.values(), iterm2.values()):
                assert_equal(item11, item22)
        else:
            assert item1 == iterm2


    assert_equal(config, old_config)
