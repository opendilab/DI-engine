import pytest
import random
import pickle
import torch

import os
import tempfile
import unittest
from unittest.mock import patch
from shutil import move

from ding.utils.file_helper import read_file, read_from_file, remove_file, save_file, read_from_path, save_file_ceph, \
                                get_disk_usage_percentage, torch_save


@pytest.mark.unittest
def test_get_disk_usage_percentage():
    assert get_disk_usage_percentage(".") >= 0.0 and get_disk_usage_percentage(".") <= 100.0
    assert get_disk_usage_percentage("./") >= 0.0 and get_disk_usage_percentage("./") <= 100.0
    assert get_disk_usage_percentage("./a/b/c") >= 0.0 and get_disk_usage_percentage("./a/b/c") <= 100.0
    assert get_disk_usage_percentage("a/b/c") >= 0.0 and get_disk_usage_percentage("a/b/c") <= 100.0


@pytest.mark.unittest
class TorchSaveTestCase(unittest.TestCase):

    def setUp(self):
        self.data = [1, 2, 3]
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.temp_dir.name, 'test_data.pt')

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_torch_save_no_retry(self):
        # Call the function without retry and skip_when_disk_full
        torch_save(self.data, self.path)

        # Assert that the file is saved
        self.assertTrue(os.path.exists(self.path))

    def test_torch_save_with_retry_failed(self):
        # Set up mock for shutil.move to simulate disk full error
        with patch('shutil.move', side_effect=OSError(28, 'No space left on device')):
            # Call the function with retry and not skip_when_disk_full
            with self.assertRaises(OSError):
                torch_save(self.data, self.path, retry=2, skip_when_disk_full=False, retry_time_interval=0.01)

        # Assert that the file is not saved
        self.assertFalse(os.path.exists(self.path))

    def test_torch_save_with_retry_failed_but_skip(self):
        # Set up mock for shutil.move to simulate disk full error but skipped
        with patch('shutil.move', side_effect=OSError(28, 'No space left on device')):
            # Call the function with retry and skip_when_disk_full
            torch_save(self.data, self.path, retry=2, skip_when_disk_full=True, retry_time_interval=0.01)

        # Assert that the file is not saved
        self.assertFalse(os.path.exists(self.path))

    def test_torch_save_with_retry_success_and_not_skip(self):
        # Set up mock for shutil.move to simulate successful move
        with patch('shutil.move', side_effect=lambda src, dst: move(src, dst)):
            # Call the function with retry and skip_when_disk_full
            torch_save(self.data, self.path, retry=2, skip_when_disk_full=False, retry_time_interval=0.01)

        # Assert that the file is saved
        self.assertTrue(os.path.exists(self.path))

    def test_torch_save_with_retry_success_and_skip(self):
        # Set up mock for shutil.move to simulate successful move
        with patch('shutil.move', side_effect=lambda src, dst: move(src, dst)):
            # Call the function with retry and skip_when_disk_full
            torch_save(self.data, self.path, retry=2, skip_when_disk_full=True, retry_time_interval=0.01)

        # Assert that the file is saved
        self.assertTrue(os.path.exists(self.path))

    def test_torch_save_non_oserror_exception(self):
        # Set up mock for shutil.move to simulate a non-OSError exception
        with patch('shutil.move', side_effect=ValueError('Some error occurred')):
            # Call the function with retry and skip_when_disk_full
            with self.assertRaises(ValueError):
                torch_save(self.data, self.path, retry=0, skip_when_disk_full=True, retry_time_interval=0.01)

        # Assert that the file is not saved
        self.assertFalse(os.path.exists(self.path))

    def test_torch_save_oserror_with_different_errno(self):
        # Set up mock for shutil.move to simulate OSError with a different errno
        with patch('shutil.move', side_effect=OSError(13, 'Permission denied')):
            # Call the function with retry and skip_when_disk_full
            with self.assertRaises(OSError):
                torch_save(self.data, self.path, retry=0, skip_when_disk_full=True, retry_time_interval=0.01)

        # Assert that the file is not saved
        self.assertFalse(os.path.exists(self.path))


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

    # test disk_avail_storage_preserve_percent
    save_file('./f_not_exist', data1, disk_avail_storage_preserve_percent=-1.0)
    assert not os.path.exists('./f_not_exist')

    remove_file("./f")
    remove_file("./f1")
    remove_file("./f2")
    remove_file("./f3")
    remove_file('./f.lock')
    remove_file('./f2.lock')
    remove_file('./f3.lock')
    remove_file('./name.txt')
