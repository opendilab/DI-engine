import easydict
import pytest
from ding.worker.learner import register_learner_hook, build_learner_hook_by_cfg, LearnerHook
from ding.worker.learner.learner_hook import SaveCkptHook, LoadCkptHook, LogShowHook, LogReduceHook
from ding.worker.learner.learner_hook import show_hooks, add_learner_hook, merge_hooks
from easydict import EasyDict


@pytest.fixture(scope='function')
def setup_simplified_hook_cfg():
    return dict(
        save_ckpt_after_iter=20,
        save_ckpt_after_run=True,
    )


@pytest.fixture(scope='function')
def fake_setup_simplified_hook_cfg():
    return dict(
        log_show_after_iter=20,
        log_reduce_after_iter=True,
    )


@pytest.mark.unittest
class TestLearnerHook:

    def test_register(self):

        class FakeHook(LearnerHook):
            pass

        register_learner_hook('fake', FakeHook)
        with pytest.raises(AssertionError):
            register_learner_hook('placeholder', type)

    def test_build_learner_hook_by_cfg(self, setup_simplified_hook_cfg):
        hooks = build_learner_hook_by_cfg(setup_simplified_hook_cfg)
        show_hooks(hooks)
        assert len(hooks['before_run']) == 0
        assert len(hooks['before_iter']) == 0
        assert len(hooks['after_iter']) == 1
        assert isinstance(hooks['after_iter'][0], SaveCkptHook)
        assert len(hooks['after_run']) == 1
        assert isinstance(hooks['after_run'][0], SaveCkptHook)

    def test_add_learner_hook(self, setup_simplified_hook_cfg):
        hooks = build_learner_hook_by_cfg(setup_simplified_hook_cfg)
        hook_1 = LogShowHook('log_show', 20, position='after_iter', ext_args=EasyDict({'freq': 100}))
        add_learner_hook(hooks, hook_1)
        hook_2 = LoadCkptHook('load_ckpt', 20, position='before_run', ext_args=EasyDict({'load_path': './model.pth'}))
        add_learner_hook(hooks, hook_2)
        hook_3 = LogReduceHook('log_reduce', 10, position='after_iter')
        add_learner_hook(hooks, hook_3)

        show_hooks(hooks)
        assert len(hooks['after_iter']) == 3
        assert len(hooks['after_run']) == 1
        assert len(hooks['before_run']) == 1
        assert len(hooks['before_iter']) == 0
        assert isinstance(hooks['after_run'][0], SaveCkptHook)
        assert isinstance(hooks['before_run'][0], LoadCkptHook)

    def test_merge_hooks(self, setup_simplified_hook_cfg, fake_setup_simplified_hook_cfg):
        hooks = build_learner_hook_by_cfg(setup_simplified_hook_cfg)
        show_hooks(hooks)
        fake_hooks = build_learner_hook_by_cfg(fake_setup_simplified_hook_cfg)
        show_hooks(fake_hooks)
        hooks_ = merge_hooks(hooks, fake_hooks)
        show_hooks(hooks_)
        assert len(hooks_['after_iter']) == 3
        assert len(hooks_['after_run']) == 1
        assert len(hooks_['before_run']) == 0
        assert len(hooks_['before_iter']) == 0
        assert isinstance(hooks['after_run'][0], SaveCkptHook)
