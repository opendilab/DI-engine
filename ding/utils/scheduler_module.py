from typing import ClassVar
from .default_helper import deep_merge_dicts
from easydict import EasyDict

class Scheduler(object):
    """
    Overview:
        Update learning parameters when the trueskill metrics has stopped improving. 
        For example, models often benefits from reducing entropy weight once the learning process stagnates.
        This scheduler reads a metrics quantity and if no improvement is seen for a 'patience' number of epochs,
        the corresponding parameter is increased or decreased, which decides on the 'schedule_mode'.
    Args:
        - schedule_flag (:obj:`bool`): Indicates whether to use scheduler in training pipeline.
            Default: False
        - schedule_parameter (:obj:`str`): One of 'entropy_weight', 'learning_rate'. 
            Default: 'entropy_weight'
        - schedule_mode (:obj:`str`): One of 'reduce', 'add','multi','div'. The schecule_mode 
            decides the way of updating the parameters.  Default:'reduce'.
        - factor (:obj:`float`) : Amount (greater than 0) by which the parameter will be 
            increased/decreased. Default: 0.05
        - change_range (:obj:`list`): Indicates the minimum and maximum value 
            the parameter can reach respectively. Default: [-1,1]
        - threshold (:obj:`float`): Threshold for measuring the new optimum,
            to only focus on significant changes. Default:  1e-4.
        - optimize_mode (:obj:`str`): One of 'min', 'max', which indicates the sign of 
            optimization objective. Dynamic_threshold = last_metrics + threshold in `max` 
            mode or last_metrics - threshold in `min` mode. Default: 'min'
        - patience (:obj:`int`): Number of epochs with no improvement after which 
            the parameter will be updated. For example, if `patience = 2`, then we 
            will ignore the first 2 epochs with no improvement, and will only update 
            the parameter after the 3rd epoch if the metrics still hasn't improved then. 
            Default: 10.
        - cooldown (:obj:`int`): Number of epochs to wait before resuming
            normal operation after the parameter has been updated. Default: 0.
    Interfaces:
        __init__, update_para, step
    Property:
        in_cooldown, is_better
    """
    
    config = dict(
                schedule_flag = False,
                schedule_parameter = 'entropy_weight',
                schedule_mode = 'reduce',
                factor = 0.05,
                change_range = [-1,1],
                threshold = 1e-4,
                optimize_mode = 'min',
                patience = 10,
                cooldown = 0, )

    def __init__(self, policy_config: EasyDict, merged_scheduler_config: EasyDict) -> None:
        '''
        Overview:
            Initialize the scheduler.
        Args:
            - policy_cfg (:obj:`EasyDict`): the config of the policy
            - merged_scheduler_config (:obj:`EasyDict`): the scheduler config, which merges the user
                config and defaul config
        '''
        schedule_parameter = merged_scheduler_config.schedule_parameter
        schedule_mode = merged_scheduler_config.schedule_mode
        factor = merged_scheduler_config.factor
        change_range = merged_scheduler_config.change_range
        threshold = merged_scheduler_config.threshold
        optimize_mode = merged_scheduler_config.optimize_mode
        patience = merged_scheduler_config.patience
        cooldown = merged_scheduler_config.cooldown

        assert schedule_parameter in ['entropy_weight', 'learning_rate'], 'The parameter to be scheduled should be one of [\'entropy_weight\', \'learning_rate\']'
        self.schedule_parameter = schedule_parameter
        
        assert schedule_mode in ['reduce','add','multi','div'], 'The schedule mode should be one of [\'reduce\', \'add\', \'multi\',\'div\']'
        self.schedule_mode = schedule_mode
        
        assert isinstance(factor, (float,int)), 'The factor should be a float/int number greater than 0'
        assert factor > 0, 'The factor should be a float/int number greater than 0'
        self.factor = float(factor)

        assert isinstance(change_range, list) and len(change_range) == 2, 'The change_range should be a list with 2 float numbers'
        assert (isinstance(change_range[0], (float,int)) ) and (isinstance(change_range[1], (float,int)) ), 'The change_range should be a list with 2 float/int numbers'
        if schedule_parameter == 'entropy_weight':
            assert (change_range[0] <= policy_config.learn.entropy_weight 
                    and change_range[1] >= policy_config.learn.entropy_weight), 'The change range should cover the initial entropy weight value'
        if schedule_parameter == 'learning_rate':
            assert (change_range[0] <= policy_config.learn.learning_rate 
                    and change_range[1] >= policy_config.learn.learning_rate), 'The change range should cover the initial learning rate value'
        self.change_range = change_range

        assert isinstance(threshold, (float,int)), 'The threshold should be a float/int number'
        self.threshold = threshold

        assert optimize_mode in ['min', 'max'], 'The optimize_mode should be one of [\'min\', \'max\']'
        self.optimize_mode = optimize_mode

        assert isinstance(patience, int), 'The patience should be a integer greater than or equal to 0'
        assert patience >= 0, 'The patience should be a integer greater than or equal to 0'
        self.patience = patience

        assert isinstance(cooldown, int), 'The cooldown_counter should be a integer greater than or equal to 0'
        assert cooldown >= 0, 'The cooldown_counter should be a integer greater than or equal to 0'
        self.cooldown = cooldown
        self.cooldown_counter = cooldown

        self.last_metrics = None
        self.bad_epochs_num = 0
        

    def step(self, metrics: float, policy_cfg: EasyDict) -> None:
        '''
        Overview:
            Decides whether to update the scheduled parameter
        Args:
            - metrics (:obj:`float`): current input metrics
            - policy_cfg (:obj:`EasyDict`): the config of the policy
        '''
        assert isinstance(metrics, float), 'The metrics should be a float number'
        cur_metrics = metrics
        
        if self.is_better(cur_metrics):
            self.bad_epochs_num = 0
        else:
            self.bad_epochs_num += 1
        self.last_metrics = cur_metrics
        
        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.bad_epochs_num = 0  # ignore any bad epochs in cooldown
        
        if self.bad_epochs_num > self.patience:
            self.update_para(policy_cfg)
            self.cooldown_counter = self.cooldown
            self.bad_epochs_num = 0
            
    
    def update_para(self, policy_cfg: EasyDict) -> None:
        '''
        Overview:
            update the scheduling parameter 
        Args:
            - policy_cfg (:obj:`EasyDict`): the config of the policy
        '''
        schedule_fn = {
                'reduce': lambda x, y, z: max(x - y, z[0]),
                'add': lambda x, y, z: min(x + y, z[1]),
                'multi': lambda x, y, z: min(x * y, z[1]) if y >= 1 else max(x * y, z[0]),
                'div': lambda x, y, z: max(x / y, z[0]) if y >= 1 else min(x / y, z[1]),
                }
            
        schedule_mode_list = list(schedule_fn.keys())

        if self.schedule_parameter == 'entropy_weight':
            if self.schedule_mode in schedule_mode_list:
                policy_cfg.learn.entropy_weight = schedule_fn(policy_cfg.learn.entropy_weight, self.factor, self.change_range)
            else:
                raise KeyError("invalid schedule_mode({}) in {}".format(self.schedule_mode, schedule_mode_list))
            

        if self.schedule_parameter == 'learning_rate':
            if self.schedule_mode in schedule_mode_list:
                policy_cfg.learn.learning_rate = schedule_fn(policy_cfg.learn.learning_rate, self.factor, self.change_range)
            else:
                raise KeyError("invalid schedule_mode({}) in {}".format(self.schedule_mode, schedule_mode_list))
            
    @property
    def in_cooldown(self) -> bool:
        '''
        Overview:
            Checks whether the scheduler is in cooldown peried. If in cooldown, the scheduler
            will ignore any bad epochs.
        '''
        return self.cooldown_counter > 0

    def is_better(self, cur: float) -> bool:
        '''
        Overview:
            Checks whether the current metrics is better than last matric with respect to threshold.
        Args:
            - cur (:obj:`float`): current metrics
        '''
        if self.last_metrics is None:
            return True

        elif self.optimize_mode == 'min':
            return cur < self.last_metrics - self.threshold
        
        elif self.optimize_mode == 'max':
            return cur > self.last_metrics + self.threshold






