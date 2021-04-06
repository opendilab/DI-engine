from easydict import EasyDict

buffer_manager_default_config = dict(
    replay_buffer=dict(
        # Max length of the buffer.
        replay_buffer_size=4096,
        # start training data count
        replay_start_size=0,
        # Max use times of one data in the buffer. Data will be removed once used for too many times.
        max_use=float("inf"),
        # Max staleness time duration of one data in the buffer; Data will be removed if
        # the duration from collecting to training is too long, i.e. The data is too stale.
        max_staleness=float("inf"),
        # Min ratio of "data count in buffer" / "sample count". If ratio is less than this, sample will return None.
        min_sample_ratio=1.,
        # (Float type) How much prioritization is used: 0 means no prioritization while 1 means full prioritization
        alpha=0.6,
        # (Float type)  How much correction is used: 0 means no correction while 1 means full correction
        beta=0.4,
        # Anneal step for beta: 0 means no annealing
        anneal_step=int(1e5),
        # Whether to track the used data or not. Buffer will use a new data structure to track data if set True.
        enable_track_used_data=False,
        # Whether to deepcopy data when willing to insert and sample data. For security purpose.
        deepcopy=False,
        # Monitor configuration for monitor and logger to use. This part does not affect buffer's function.
        monitor=dict(
            # Frequency of logging
            log_freq=2000,
            # Logger's save path
            log_path='./log/buffer/agent_buffer/',
            # Natural time expiration. Used for log data smoothing.
            natural_expire=100,
            # Tick time expiration. Used for log data smoothing.
            tick_expire=100,
        ),
    ),
)
buffer_manager_default_config = EasyDict(buffer_manager_default_config)
