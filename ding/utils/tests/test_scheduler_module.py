from easydict import EasyDict
import pytest
from ding.utils.scheduler_module import Scheduler
from dizoo.league_demo.league_demo_ppo_config import league_demo_ppo_config


@pytest.mark.unittest
class TestSchedulerModule():

    test_merged_scheduler_config = dict(
        schedule_flag=False,
        schedule_parameter='entropy_weight',
        schedule_mode='reduce',
        factor=0.05,
        change_range=[-1, 1],
        threshold=1e-4,
        optimize_mode='min',
        patience=1,
        cooldown=0,
    )
    test_merged_scheduler_config = EasyDict(test_merged_scheduler_config)
    test_policy_config = EasyDict(league_demo_ppo_config.policy)

    def test_init_schedule_parameter(self):
        self.test_merged_scheduler_config.schedule_parameter = 'clip_ratio'
        with pytest.raises(AssertionError) as excinfo:
            test_scheduler = Scheduler(self.test_policy_config, self.test_merged_scheduler_config)
        assert 'entropy_weight' in str(excinfo.value)

        # recover the correct value for later test function
        self.test_merged_scheduler_config.schedule_parameter = 'entropy_weight'

    def test_init_factor(self):
        self.test_merged_scheduler_config.factor = 'hello_test'
        with pytest.raises(AssertionError) as excinfo:
            test_scheduler = Scheduler(self.test_policy_config, self.test_merged_scheduler_config)
        assert 'float/int' in str(excinfo.value)

        self.test_merged_scheduler_config.factor = 0
        with pytest.raises(AssertionError) as excinfo:
            test_scheduler = Scheduler(self.test_policy_config, self.test_merged_scheduler_config)
        assert 'greater than 0' in str(excinfo.value)

        # recover the correct value for later test function
        self.test_merged_scheduler_config.factor = 0.05

    def test_init_change_range(self):
        self.test_merged_scheduler_config.change_range = 0
        with pytest.raises(AssertionError) as excinfo:
            test_scheduler = Scheduler(self.test_policy_config, self.test_merged_scheduler_config)
        assert 'list' in str(excinfo.value)

        self.test_merged_scheduler_config.change_range = [0, 'hello_test']
        with pytest.raises(AssertionError) as excinfo:
            test_scheduler = Scheduler(self.test_policy_config, self.test_merged_scheduler_config)
        assert 'float' in str(excinfo.value)

        self.test_merged_scheduler_config.change_range = [0, -1]
        with pytest.raises(AssertionError) as excinfo:
            test_scheduler = Scheduler(self.test_policy_config, self.test_merged_scheduler_config)
        assert 'smaller' in str(excinfo.value)

        # recover the correct value for later test function
        self.test_merged_scheduler_config.change_range = [-1, 1]

    def test_is_better(self):
        test_scheduler = Scheduler(self.test_policy_config, self.test_merged_scheduler_config)
        assert test_scheduler.is_better(-1) is True

        test_scheduler.last_metrics = 1
        assert test_scheduler.is_better(0.5) is True

    def test_in_cooldown(self):
        self.test_merged_scheduler_config.cooldown_counter = 0
        test_scheduler = Scheduler(self.test_policy_config, self.test_merged_scheduler_config)
        assert test_scheduler.in_cooldown is False

    def test_step(self):

        self.test_merged_scheduler_config.cooldown = 1

        test_scheduler = Scheduler(self.test_policy_config, self.test_merged_scheduler_config)
        assert test_scheduler.cooldown_counter == 1
        test_scheduler.last_metrics = 1.0

        old_weight = self.test_policy_config.learn.entropy_weight

        # good epoch with maximum cooldown lenth is 1
        test_scheduler.step(0.9, self.test_policy_config)
        assert self.test_policy_config.learn.entropy_weight == old_weight
        assert test_scheduler.cooldown_counter == 0
        assert test_scheduler.last_metrics == 0.9
        assert test_scheduler.bad_epochs_num == 0

        # first bad epoch in cooldown period
        test_scheduler.step(0.899999, self.test_policy_config)
        assert self.test_policy_config.learn.entropy_weight == old_weight
        assert test_scheduler.cooldown_counter == 0
        assert test_scheduler.last_metrics == 0.899999
        assert test_scheduler.bad_epochs_num == 1

        # first bad epoch after cooldown
        test_scheduler.step(0.899998, self.test_policy_config)
        assert self.test_policy_config.learn.entropy_weight == old_weight - self.test_merged_scheduler_config.factor
        assert test_scheduler.cooldown_counter == 1
        assert test_scheduler.last_metrics == 0.899998
        assert test_scheduler.bad_epochs_num == 0
