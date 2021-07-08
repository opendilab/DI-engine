import pytest
import random
import pickle

from ding.utils.file_helper import read_file, read_from_file, remove_file, save_file, read_from_path, save_file_ceph


@pytest.mark.unittest
def test_normal_file():
    data1 = {'a': [random.randint(0, 100) for i in range(100)]}
    save_file('./f', data1)
    data2 = read_file("./f")
    assert (data2 == data1)
    with open("./f1", "wb") as f1:
        pickle.dump(data1, f1)
    data3 = read_from_file("./f1")
    assert (data3 == data1)
    data4 = read_from_path("./f1")
    assert (data4 == data1)
    save_file_ceph("./f2", data1)
    assert (data1 == read_from_file("./f2"))
    # test lock
    save_file('./f3', data1, use_lock=True)
    data_read = read_file('./f3', use_lock=True)
    assert isinstance(data_read, dict)

    remove_file("./f")
    remove_file("./f1")
    remove_file("./f2")
    remove_file("./f3")
    remove_file('./f.lock')
    remove_file('./f2.lock')
    remove_file('./f3.lock')
    remove_file('./name.txt')
