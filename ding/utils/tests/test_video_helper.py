import os
import shutil
import tempfile
import pytest
import numpy as np
from ding.utils.video_helper import numpy_array_to_video

@pytest.mark.unittest
class TestVideoHelper:

    def test_numpy_array_to_video(self):

        # create a numpy array
        frames = np.random.randint(0, 255, size=(100, 100, 100, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_file = os.path.join(tmpdir, 'temp_file.mp4')
            numpy_array_to_video(frames, temp_file, fps=30.0)
            assert os.path.exists(temp_file)
