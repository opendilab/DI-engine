import shutil
import tempfile
from time import sleep, time
import pytest
from ding.data.model_loader import FileModelLoader
from ding.data.storage.file import FileModelStorage
from ding.model import DQN
from ding.config import compile_config
from dizoo.atari.config.serial.pong.pong_dqn_config import main_config, create_config
from os import path


@pytest.mark.unittest
def test_model_loader():
    tempdir = path.join(tempfile.gettempdir(), "test_model_loader")
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    model = DQN(**cfg.policy.model)
    loader = FileModelLoader(model=model, dirname=tempdir)
    try:
        loader.start()
        model_storage = None

        def save_model(storage):
            nonlocal model_storage
            model_storage = storage

        start = time()
        loader.save(save_model)
        save_time = time() - start
        print("Save time: {:.4f}s".format(save_time))
        assert save_time < 0.01
        sleep(1)
        assert isinstance(model_storage, FileModelStorage)

        state_dict = loader.load(model_storage)
        model.load_state_dict(state_dict)
    finally:
        if path.exists(tempdir):
            shutil.rmtree(tempdir)
        loader.shutdown()


@pytest.mark.benchmark
def test_model_loader_benchmark():
    pass
