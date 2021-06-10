from easydict import EasyDict

conv1d_config = dict(
    Feature_Enbedding = dict(
        Player=dict(
            input_dim = 36,
            output_dim = 64,
        ),
        Ball=dict(
            input_dim = 18,
            output_dim = 64,
        ),
        LeftTeam=dict(
            input_dim = 7,
            output_dim = 48,
            conv1d_output_channel = 36,
            fc_output_dim = 96,
        ),
        RightTeam=dict(
            input_dim = 7,
            output_dim = 48,
            conv1d_output_channel = 36,
            fc_output_dim = 96,
        ),
        LeftClosest=dict(
            input_dim= 7,
            output_dim=48,
        ),
        RightClosest=dict(
            input_dim=7,
            output_dim = 48,
        )
    ),
    FC_CAT = dict(
        input_dim = 416,
    ),
    LSTM_size = 256,
    Policy_Head = dict(
        input_dim = 256,
        hidden_dim = 164,
        act_shape = 19,
    ),
    Value_Head = dict(
        input_dim = 256,
        hidden_dim = 164,
        output_dim = 1
    ),

)

conv1d_default_config = EasyDict(conv1d_config)
main_config = conv1d_default_config
create_config = dict()