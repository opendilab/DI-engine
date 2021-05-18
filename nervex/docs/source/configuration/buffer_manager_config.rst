Buffer Manager Config
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    buffer_manager_default_config = dict(
        replay_buffer=dict(
            # First list all buffers' names; Then they show up as keys and
            # their correspoing configurations are their values.
            buffer_name=['agent', 'demo'],
            # Buffer1 is called "agent". You can regard this as a most general and frequently used buffer config.
            agent=dict(
                # Max length of the buffer.
                replay_buffer_size=4096,
                # Max use times of one data in the buffer. Data will be removed once used for too many times.
                max_use=16,
                # Max staleness time duration of one data in the buffer; Data will be removed if
                # the duration from collecting to training is too long, i.e. The data is too stale.
                max_staleness=10000,
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
            # Buffer2 is called "demo". It is a demonstration buffer proposed in R2D3. It shares many common things with
            # common "agent" buffer except loading expert data at the beginning. That's why it has a "load_path" key.
            demo=dict(
                replay_buffer_size=4096,
                max_use=16,
                max_staleness=10000,
                alpha=0.6,
                beta=0.4,
                anneal_step=0,
                enable_track_used_data=False,
                deepcopy=False,
                monitor=dict(
                    log_freq=2000,
                    log_path='./log/buffer/demo_buffer/',
                    natural_expire=100,
                    tick_expire=100,
                ),
            ),
            # The ratio for sampling from different buffers. You must guarantee that all ratio's sum is 1.
            sample_ratio=dict(
                agent=0.99609375,
                demo=0.00390625,
            ),
        ),
    )
