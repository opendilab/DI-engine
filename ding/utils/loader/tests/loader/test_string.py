import re

import pytest

from ding.utils.loader import enum, rematch, regrep, to_type


@pytest.mark.unittest
class TestConfigLoaderString:

    def test_enum_plain(self):
        _loader = enum('red', 'green', 'blue', 'yellow')
        assert _loader('red') == 'red'
        assert _loader('green') == 'green'
        assert _loader('blue') == 'blue'
        assert _loader('yellow') == 'yellow'
        with pytest.raises(ValueError):
            _loader(int)
        with pytest.raises(ValueError):
            _loader('Red')
        with pytest.raises(ValueError):
            _loader('YELLOW')
        with pytest.raises(ValueError):
            _loader(1)
        with pytest.raises(ValueError):
            _loader(None)

    def test_enum_case_insensitive(self):
        _loader = enum('red', 'green', 'blue', 'yellow', case_sensitive=False)
        assert _loader('red') == 'red'
        assert _loader('green') == 'green'
        assert _loader('blue') == 'blue'
        assert _loader('yellow') == 'yellow'
        with pytest.raises(ValueError):
            _loader(int)
        assert _loader('Red') == 'red'
        assert _loader('YELLOW') == 'yellow'
        with pytest.raises(ValueError):
            _loader(1)
        with pytest.raises(ValueError):
            _loader(None)

    def test_enum_complex_case_1(self):
        _loader = (lambda x: str(x).strip()) >> enum('red', 'green', 'blue', 'yellow', case_sensitive=False)
        assert _loader('red') == 'red'
        assert _loader('green') == 'green'
        assert _loader('blue') == 'blue'
        assert _loader('yellow') == 'yellow'
        assert _loader(' yellow ') == 'yellow'
        with pytest.raises(ValueError):
            _loader(int)
        assert _loader('Red') == 'red'
        assert _loader('YELLOW') == 'yellow'
        assert _loader(' YelloW ') == 'yellow'
        with pytest.raises(ValueError):
            _loader(1)
        with pytest.raises(ValueError):
            _loader(None)

    # noinspection DuplicatedCode
    def test_rematch_str(self):
        _loader = to_type(str) >> str.strip >> str.lower >> rematch('[0-9a-z_]+@([0-9a-z]+.)+[0-9a-z]+')
        assert _loader('hansbug@buaa.edu.cn') == 'hansbug@buaa.edu.cn'
        assert _loader(' hansbug@BUAA.EDU.CN\t') == 'hansbug@buaa.edu.cn'
        with pytest.raises(ValueError):
            _loader(' hansbug.buaa.edu.cn')
        with pytest.raises(ValueError):
            _loader(' hansbug@cn')
        with pytest.raises(ValueError):
            _loader(' hansbug@buaa.edu..cn')

    # noinspection DuplicatedCode
    def test_rematch_pattern(self):
        _loader = to_type(str) >> str.strip >> str.lower >> rematch(re.compile('[0-9a-z_]+@([0-9a-z]+.)+[0-9a-z]+'))
        assert _loader('hansbug@buaa.edu.cn') == 'hansbug@buaa.edu.cn'
        assert _loader(' hansbug@BUAA.EDU.CN\t') == 'hansbug@buaa.edu.cn'
        with pytest.raises(ValueError):
            _loader(' hansbug.buaa.edu.cn')
        with pytest.raises(ValueError):
            _loader(' hansbug@cn')
        with pytest.raises(ValueError):
            _loader(' hansbug@buaa.edu..cn')

    def test_rematch_invalid(self):
        with pytest.raises(TypeError):
            _loader = rematch(1)

    def test_regrep(self):
        _loader = to_type(str) >> str.lower >> regrep('[0-9a-z_]+@([0-9a-z]+.)+[0-9a-z]+')
        assert _loader('hansbug@buaa.edu.cn') == 'hansbug@buaa.edu.cn'
        assert _loader(' hansbug@BUAA.EDU.CN\t') == 'hansbug@buaa.edu.cn'
        assert _loader('This is my email hansbug@buaa.edu.cn, thanks~~') == 'hansbug@buaa.edu.cn'
        with pytest.raises(ValueError):
            _loader('this is hansbug.buaa.edu.cn')

    def test_regrep_group(self):
        _loader = to_type(str) >> str.lower >> regrep('[0-9a-z_]+@(([0-9a-z]+.)+[0-9a-z]+)', group=1)
        assert _loader('hansbug@buaa.edu.cn') == 'buaa.edu.cn'
        assert _loader(' hansbug@BUAA.EDU.CN\t') == 'buaa.edu.cn'
        assert _loader('This is my email hansbug@buaa.edu.cn, thanks~~') == 'buaa.edu.cn'
        with pytest.raises(ValueError):
            _loader(' @buaa.edu.cn')
