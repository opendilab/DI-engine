from easydict import EasyDict
import pytest
import unittest
from unittest import mock
from unittest.mock import patch
import pathlib as pl
import os, shutil

from ding.utils.profiler_helper import Profiler, regist_profiler


@pytest.mark.unittest
class TestProfilerModule:

    def assertIsFile(self, path):
        if not pl.Path(path).resolve().is_file():
            raise AssertionError("File does not exist: %s" % str(path))

    def test(self):
        profiler = Profiler()

        def register_mock(write_profile, pr, folder_path):
            profiler.write_profile(pr, folder_path)

        def clean_up(dir):
            if os.path.exists(dir):
                shutil.rmtree(dir)

        dir = "./tmp_test/"
        clean_up(dir)

        with patch('ding.utils.profiler_helper.regist_profiler', register_mock):
            profiler.profile(dir)
            file_path = os.path.join(dir, "profile_tottime.txt")
            self.assertIsFile(file_path)
            file_path = os.path.join(dir, "profile_cumtime.txt")
            self.assertIsFile(file_path)
            file_path = os.path.join(dir, "profile.prof")
            self.assertIsFile(file_path)
            file_path = os.path.join(dir, "profile.svg")
            self.assertIsFile(file_path)
        clean_up(dir)