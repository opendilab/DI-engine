
class Scheduler(object):
    """
    Overview:
        Update learning parameters when the trueskill metrics has stopped improving. 
        For example, models often benefits from reducing entropy weight once the learning process stagnates.
        This scheduler reads a metrics quantity and if no improvement is seen for a 'patience' number of epochs,
        the corresponding parameter is increased or decreased, which decides on the 'schedule_mode'.
    
    Args:
        schedule_parameter (str) : One of 'entropy_weight', 'learning_rate'. 
            Default: 'entropy_weight'
        schedule_mode (str) : One of 'reduce', 'add','multi'. The schecule_mode 
            decides the way of updating the parameters.  Default:'reduce'.
        factor (float) : Amount by which the parameter will be 
            increased/decreased. Default: 0.05
        change_range (list) : Indicates the minimum and maximum value 
            the parameter can reach respectively. Default: [-1,1]
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default:  1e-4.
        optimize_mode (str): One of 'min', 'max', which indicates the sign of 
            optimization objective. Dynamic_threshold = last_metrics + threshold in `max` 
            mode or last_metrics - threshold in `min` mode. Default: 'min'
        patience (int) : Number of epochs with no improvement after which 
            the parameter will be updated. For example, if `patience = 2`, then we 
            will ignore the first 2 epochs with no improvement, and will only update 
            the parameter after the 3rd epoch if the metrics still hasn't improved then. 
            Default: 10.
        cooldown (int) : Number of epochs to wait before resuming
            normal operation after the parameter has been updated. Default: 0.
    """

    def __init__(self, cfg,
                # schedule_parameter = 'entropy_weight',
                # schedule_mode = 'reduce',
                # factor = 0.05,
                # change_range = [-1,1],
                # threshold = 1e-4,
                # optimize_mode = 'min',
                # patience = 10,
                # cooldown = 0
                ):
        
        schedule_parameter = cfg.policy.scheduler.schedule_parameter
        schedule_mode = cfg.policy.scheduler.schedule_mode
        factor = cfg.policy.scheduler.factor
        change_range = cfg.policy.scheduler.change_range
        threshold = cfg.policy.scheduler.threshold
        optimize_mode = cfg.policy.scheduler.optimize_mode
        patience = cfg.policy.scheduler.patience
        cooldown = cfg.policy.scheduler.cooldown
        
        assert schedule_parameter in ['entropy_weight', 'learning_rate'], 'The parameter to be scheduled should be one of [\'entropy_weight\', \'learning_rate\']'
        self.schedule_parameter = schedule_parameter
        
        assert schedule_mode in ['reduce','add','multi'], 'The schedule mode should be one of [\'reduce\', \'add\', \'multi\']'
        self.schedule_mode = schedule_mode
        
        assert isinstance(factor, float) or isinstance(factor, int), 'The factor should be a float/int number greater than 0'
        assert factor > 0, 'The factor should be a float/int number greater than 0'
        self.factor = factor

        assert isinstance(change_range, list) and len(change_range) == 2, 'The change_range should be a list with 2 float numbers'
        assert (isinstance(change_range[0], float) or isinstance(change_range[0], int)) and (isinstance(change_range[1], float) or isinstance(change_range[1], int)), 'The change_range should be a list with 2 float/int numbers'
        if schedule_parameter == 'entropy_weight':
            assert (change_range[0] <= cfg.policy.learn.entropy_weight 
                    and change_range[1] >= cfg.policy.learn.entropy_weight), 'The change range should cover the initial entropy weight value'
        if schedule_parameter == 'learning_rate':
            assert (change_range[0] <= cfg.policy.learn.learning_rate 
                    and change_range[1] >= cfg.policy.learn.learning_rate), 'The change range should cover the initial learning rate value'
        self.change_range = change_range

        assert isinstance(threshold, float) or isinstance(threshold, int), 'The threshold should be a float/int number'
        self.threshold = threshold

        assert optimize_mode in ['min', 'max'], 'The optimize_mode should be one of [\'min\', \'max\']'
        self.optimize_mode = optimize_mode

        assert isinstance(patience, int), 'The patience should be a integer greater than or equal to 0'
        assert patience >= 0, 'The patience should be a integer greater than or equal to 0'
        self.patience = patience

        assert isinstance(cooldown, int), 'The cooldown_counter should be a integer greater than or equal to 0'
        assert cooldown >= 0, 'The cooldown_counter should be a integer greater than or equal to 0'
        self.cooldown= cooldown
        self.cooldown_counter = cooldown

        self.last_metrics = None
        self.bad_epochs_num = 0
        

    def step(self, metrics, cfg):
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
            self.update_para(cfg)
            self.cooldown_counter = self.cooldown
            self.bad_epochs_num = 0
    
    def update_para(self, cfg):
        if self.schedule_parameter == 'entropy_weight':
            entropy_weight = cfg.policy.learn.entropy_weight
            if self.schedule_mode == 'reduce':
                cfg.policy.learn.entropy_weight = max(entropy_weight - self.factor, self.change_range[0])
            
            if self.schedule_mode == 'add':
                cfg.policy.learn.entropy_weight = min(entropy_weight + self.factor, self.change_range[1])
            
            if self.schedule_mode == 'multi':
                if self.factor >= 1:
                    cfg.policy.learn.entropy_weight = min (self.factor*entropy_weight, self.change_range[1])
                else:
                    cfg.policy.learn.entropy_weight = max (self.factor*entropy_weight, self.change_range[0])

        if self.schedule_parameter == 'learning_rate':
            learning_rate = cfg.policy.learn.learning_rate
            if self.schedule_mode == 'reduce':
                cfg.policy.learn.learning_rate = max(learning_rate - self.factor, self.change_range[0])
                
            if self.schedule_mode == 'add':
                cfg.policy.learn.learning_rate = min(learning_rate + self.factor, self.change_range[1])

            if self.schedule_mode == 'multi':
                if self.factor >= 1:
                    cfg.policy.learn.learning_rate = min (self.factor*learning_rate, self.change_range[1])
                else:
                    cfg.policy.learn.learning_rate = max (self.factor*learning_rate, self.change_range[0])

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0
    
    def is_better(self, cur):
        if self.last_metrics == None:
            return True

        if self.optimize_mode == 'min':
            return cur < self.last_metrics - self.threshold
        
        if self.optimize_mode == 'max':
            return cur > self.last_metrics + self.threshold






