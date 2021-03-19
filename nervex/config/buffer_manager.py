from easydict import EasyDict

buffer_manager_default_config = dict(
    replay_buffer=dict(
        # First list all buffers' names; Then they show up as keys and
        # their correspoing configurations are their values.
        buffer_name=['agent'],
        # Buffer1 is called "agent". You can regard this as a most general and frequently used buffer config.
        agent=dict(
            # Max length of the buffer.
            meta_maxlen=4096,
            # Max reuse times of one data in the buffer. Data will be removed once reused for too many times.
            max_reuse=16,
            # Max staleness time duration of one data in the buffer; Data will be removed if
            # the duration from collecting to training is too long, i.e. The data is too stale.
            max_staleness=10000,
            # Min ratio of "data count in buffer" / "sample count". If ratio is less than this, sample will return None.
            min_sample_ratio=1.,
            # (Float type) How much prioritization is used: 0 means no prioritization while 1 means full prioritization
            alpha=0.6,
            # (Float type)  How much correction is used: 0 means no correction while 1 means full correction
            beta=0.4,
            # Anneal step for beta: 0 means no annealing
            anneal_step=0,
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
    ),
)
buffer_manager_default_config = EasyDict(buffer_manager_default_config)
