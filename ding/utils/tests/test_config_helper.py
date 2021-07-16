import pytest
import os
import copy
from easydict import EasyDict

from ding.config import read_config_directly, save_config
from ding.utils.default_helper import deep_merge_dicts, flatten_dict, deep_update


@pytest.mark.unittest
class TestConfigHelper():

    def test_flatten_dict(self):
        dict1 = {'a': {'aa': {'aaa': 'data - aaa'}, 'ab': 'data - ab'}}
        dict2 = {'a/ab': 'data - ab', 'a/aa/aaa': 'data - aaa'}
        assert flatten_dict(dict1) == dict2

    def test_deep_merge_dicts(self):
        dict1 = {'a': {'aa': 'aa1', 'ab': 'ab2'}, 'b': {'bb': 'bb2'}}
        dict2 = {'a': {'aa': 'aa2', 'ac': 'ab1'}, 'b': {'ba': 'ba2'}, 'c': {}}
        merged = {'a': {'aa': 'aa2', 'ab': 'ab2', 'ac': 'ab1'}, 'b': {'bb': 'bb2', 'ba': 'ba2'}, 'c': {}}
        assert deep_merge_dicts(dict1, dict2) == merged
        with pytest.raises(RuntimeError):
            deep_update(dict1, dict2, new_keys_allowed=False)

    def test_config(self):
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
        cfg_path = tempfile.mktemp(suffix=".py")
        save_config(old_config, cfg_path)
        assert os.path.exists(cfg_path)
        config = read_config_directly(cfg_path)["exp_config"]

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
