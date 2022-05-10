from collections import namedtuple

import numpy as np
import pytest
import torch

from ding.utils.default_helper import lists_to_dicts, dicts_to_lists, squeeze, default_get, override, error_wrapper, \
    list_split, LimitedSpaceContainer, set_pkg_seed, deep_merge_dicts, deep_update, flatten_dict, RunningMeanStd, \
    one_time_warning, split_data_generator


@pytest.mark.unittest
class TestDefaultHelper():

    def test_lists_to_dicts(self):
        set_pkg_seed(12)
        with pytest.raises(ValueError):
            lists_to_dicts([])
        with pytest.raises(TypeError):
            lists_to_dicts([1])
        assert lists_to_dicts([{1: 1, 10: 3}, {1: 2, 10: 4}]) == {1: [1, 2], 10: [3, 4]}
        T = namedtuple('T', ['location', 'race'])
        data = [T({'x': 1, 'y': 2}, 'zerg') for _ in range(3)]
        output = lists_to_dicts(data)
        assert isinstance(output, T) and output.__class__ == T
        assert len(output.location) == 3
        data = [{'value': torch.randn(1), 'obs': {'scalar': torch.randn(4)}} for _ in range(3)]
        output = lists_to_dicts(data, recursive=True)
        assert isinstance(output, dict)
        assert len(output['value']) == 3
        assert len(output['obs']['scalar']) == 3

    def test_dicts_to_lists(self):
        assert dicts_to_lists({1: [1, 2], 10: [3, 4]}) == [{1: 1, 10: 3}, {1: 2, 10: 4}]

    def test_squeeze(self):
        assert squeeze((4, )) == 4
        assert squeeze({'a': 4}) == 4
        assert squeeze([1, 3]) == (1, 3)
        data = np.random.randn(3)
        output = squeeze(data)
        assert (output == data).all()

    def test_default_get(self):
        assert default_get({}, 'a', default_value=1, judge_fn=lambda x: x < 2) == 1
        assert default_get({}, 'a', default_fn=lambda: 1, judge_fn=lambda x: x < 2) == 1
        with pytest.raises(AssertionError):
            default_get({}, 'a', default_fn=lambda: 1, judge_fn=lambda x: x < 0)
        assert default_get({'val': 1}, 'val', default_value=2) == 1

    def test_override(self):

        class foo(object):

            def fun(self):
                raise NotImplementedError

        class foo1(foo):

            @override(foo)
            def fun(self):
                return "a"

        with pytest.raises(NameError):

            class foo2(foo):

                @override(foo)
                def func(self):
                    pass

        with pytest.raises(NotImplementedError):
            foo().fun()
        foo1().fun()

    def test_error_wrapper(self):

        def good_ret(a, b=1):
            return a + b

        wrap_good_ret = error_wrapper(good_ret, 0)
        assert good_ret(1) == wrap_good_ret(1)

        def bad_ret(a, b=0):
            return a / b

        wrap_bad_ret = error_wrapper(bad_ret, 0)
        assert wrap_bad_ret(1) == 0
        wrap_bad_ret_with_customized_log = error_wrapper(bad_ret, 0, 'customized_information')

    def test_list_split(self):
        data = [i for i in range(10)]
        output, residual = list_split(data, step=4)
        assert len(output) == 2
        assert output[1] == [4, 5, 6, 7]
        assert residual == [8, 9]
        output, residual = list_split(data, step=5)
        assert len(output) == 2
        assert output[1] == [5, 6, 7, 8, 9]
        assert residual is None


@pytest.mark.unittest
class TestLimitedSpaceContainer():

    def test_container(self):
        container = LimitedSpaceContainer(0, 5)
        first = container.acquire_space()
        assert first
        assert container.cur == 1
        left = container.get_residual_space()
        assert left == 4
        assert container.cur == container.max_val == 5
        no_space = container.acquire_space()
        assert not no_space
        container.increase_space()
        six = container.acquire_space()
        assert six
        for i in range(6):
            container.release_space()
            assert container.cur == 5 - i
        container.decrease_space()
        assert container.max_val == 5


@pytest.mark.unittest
class TestDict:

    def test_deep_merge_dicts(self):
        dict1 = {
            'a': 3,
            'b': {
                'c': 3,
                'd': {
                    'e': 6,
                    'f': 5,
                }
            }
        }
        dict2 = {
            'b': {
                'c': 5,
                'd': 6,
                'g': 4,
            }
        }
        new_dict = deep_merge_dicts(dict1, dict2)
        assert new_dict['a'] == 3
        assert isinstance(new_dict['b'], dict)
        assert new_dict['b']['c'] == 5
        assert new_dict['b']['c'] == 5
        assert new_dict['b']['g'] == 4

    def test_deep_update(self):
        dict1 = {
            'a': 3,
            'b': {
                'c': 3,
                'd': {
                    'e': 6,
                    'f': 5,
                },
                'z': 4,
            }
        }
        dict2 = {
            'b': {
                'c': 5,
                'd': 6,
                'g': 4,
            }
        }
        with pytest.raises(RuntimeError):
            new1 = deep_update(dict1, dict2, new_keys_allowed=False)
        new2 = deep_update(dict1, dict2, new_keys_allowed=False, whitelist=['b'])
        assert new2['a'] == 3
        assert new2['b']['c'] == 5
        assert new2['b']['d'] == 6
        assert new2['b']['g'] == 4
        assert new2['b']['z'] == 4

        dict1 = {
            'a': 3,
            'b': {
                'type': 'old',
                'z': 4,
            }
        }
        dict2 = {
            'b': {
                'type': 'new',
                'c': 5,
            }
        }
        new3 = deep_update(dict1, dict2, new_keys_allowed=True, whitelist=[], override_all_if_type_changes=['b'])
        assert new3['a'] == 3
        assert new3['b']['type'] == 'new'
        assert new3['b']['c'] == 5
        assert 'z' not in new3['b']

    def test_flatten_dict(self):
        dict = {
            'a': 3,
            'b': {
                'c': 3,
                'd': {
                    'e': 6,
                    'f': 5,
                },
                'z': 4,
            }
        }
        flat = flatten_dict(dict)
        assert flat['a'] == 3
        assert flat['b/c'] == 3
        assert flat['b/d/e'] == 6
        assert flat['b/d/f'] == 5
        assert flat['b/z'] == 4

    def test_one_time_warning(self):
        one_time_warning('test_one_time_warning')

    def test_running_mean_std(self):
        running = RunningMeanStd()
        running.reset()
        running.update(np.arange(1, 10))
        assert running.mean == pytest.approx(5, abs=1e-4)
        assert running.std == pytest.approx(2.582030, abs=1e-6)
        running.update(np.arange(2, 11))
        assert running.mean == pytest.approx(5.5, abs=1e-4)
        assert running.std == pytest.approx(2.629981, abs=1e-6)
        running.reset()
        running.update(np.arange(1, 10))
        assert pytest.approx(running.mean, abs=1e-4) == 5
        assert running.mean == pytest.approx(5, abs=1e-4)
        assert running.std == pytest.approx(2.582030, abs=1e-6)
        new_shape = running.new_shape((2, 4), (3, ), (1, ))
        assert isinstance(new_shape, tuple) and len(new_shape) == 3

        running = RunningMeanStd(shape=(4, ))
        running.reset()
        running.update(np.random.random((10, 4)))
        assert isinstance(running.mean, torch.Tensor) and running.mean.shape == (4, )
        assert isinstance(running.std, torch.Tensor) and running.std.shape == (4, )

    def test_split_data_generator(self):

        def get_data():
            return {
                'obs': torch.randn(5),
                'action': torch.randint(0, 10, size=(1, )),
                'prev_state': [None, None],
                'info': {
                    'other_obs': torch.randn(5)
                },
            }

        data = [get_data() for _ in range(4)]
        data = lists_to_dicts(data)
        data['obs'] = torch.stack(data['obs'])
        data['action'] = torch.stack(data['action'])
        data['info'] = {'other_obs': torch.stack([t['other_obs'] for t in data['info']])}
        assert len(data['obs']) == 4
        data['NoneKey'] = None
        generator = split_data_generator(data, 3)
        generator_result = list(generator)
        assert len(generator_result) == 2
        assert generator_result[0]['NoneKey'] is None
        assert len(generator_result[0]['obs']) == 3
        assert generator_result[0]['info']['other_obs'].shape == (3, 5)
        assert generator_result[1]['NoneKey'] is None
        assert len(generator_result[1]['obs']) == 3
        assert generator_result[1]['info']['other_obs'].shape == (3, 5)

        generator = split_data_generator(data, 3, shuffle=False)
