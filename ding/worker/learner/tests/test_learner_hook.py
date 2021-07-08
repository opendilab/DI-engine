import pytest
from ding.worker.learner import register_learner_hook, build_learner_hook_by_cfg, LearnerHook
from ding.worker.learner.learner_hook import SaveCkptHook, show_hooks


@pytest.fixture(scope='function')
def setup_simplified_hook_cfg():
    return dict(
        save_ckpt_after_iter=20,
        save_ckpt_after_run=True,
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
