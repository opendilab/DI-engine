from easydict import EasyDict

conv1d_config = dict(
    feature_embedding=dict(
        player=dict(
            input_dim=36,
            output_dim=64,
        ),
        ball=dict(
            input_dim=18,
            output_dim=64,
        ),
        left_team=dict(
            input_dim=7,
            output_dim=48,
            conv1d_output_channel=36,
            fc_output_dim=96,
        ),
        right_team=dict(
            input_dim=7,
            output_dim=48,
            conv1d_output_channel=36,
            fc_output_dim=96,
        ),
        left_closest=dict(
            input_dim=7,
            output_dim=48,
        ),
        right_closest=dict(
            input_dim=7,
            output_dim=48,
        )
    ),
    fc_cat=dict(input_dim=416, ),
    lstm_size=256,
    policy_head=dict(
        input_dim=256,
        hidden_dim=164,
        act_shape=19,
    ),
    value_head=dict(input_dim=256, hidden_dim=164, output_dim=1),
)

conv1d_default_config = EasyDict(conv1d_config)
