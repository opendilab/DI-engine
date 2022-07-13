from easydict import EasyDict
from ding.rl_utils.mcts.game_base_config import GameBaseConfig, DiscreteSupport

game_config = EasyDict(dict(
    # NOTE
    env_name='PongNoFrameskip-v4',
    model_type='atari',
    device='cuda',
    # device='cpu',
    # if mcts_ctree=True, using cpp mcts code
    mcts_ctree=True,
    image_based=True,
    cvt_string=True,
    clip_reward=True,
    game_wrapper=True,
    action_space_size=6,
    amp_type='none',
    # gray_scale=False,
    # obs_shape=(12, 96, 96),
    # obs_shape=(12, 84, 84),
    # image_channel=3,
    gray_scale=True,
    image_channel=1,
    obs_shape=(4, 84, 84),

    downsample=True,
    vis_result=True,
    # TODO(pu): test the effect of augmentation
    use_augmentation=True,
    # Style of augmentation
    # choices=['none', 'rrc', 'affine', 'crop', 'blur', 'shift', 'intensity']
    augmentation=['shift', 'intensity'],

    # debug
    # collector_env_num=1,
    # evaluator_env_num=1,
    # max_episode_steps=int(100),
    # test_max_episode_steps=int(100),
    # num_simulations=2,
    # batch_size=4,
    # game_history_length=20,
    # total_transitions=int(1e5),
    # num_unroll_steps=5,
    # td_steps=5,

    collector_env_num=1,
    evaluator_env_num=3,
    max_episode_steps=int(1.08e5),
    test_max_episode_steps=int(1.08e5),
    # TODO(pu): how to set proper num_simulations?
    num_simulations=50,
    batch_size=256,
    game_history_length=400,
    total_transitions=int(1e5),
    num_unroll_steps=5,
    td_steps=5,

    # TODO(pu): use dqn expert data
    revisit_policy_search_rate=1,

    # TODO(pu): why not use adam?
    lr_manually=True,

    # TODO(pu): if true, no priority to sample
    use_max_priority=True,    # if true, sample without priority
    # use_max_priority=False,
    use_priority=True,

    # TODO(pu): only used for adjust temperature manually
    max_training_steps=int(1e5),
    auto_temperature=False,
    # only effective when auto_temperature=False
    fixed_temperature_value=0.25,
    # TODO(pu): whether to use root value in reanalyzing?
    use_root_value=False,

    # TODO(pu): test the effect
    init_zero=True,
    state_norm=False,

    mini_infer_size=2,
    # (Float type) How much prioritization is used: 0 means no prioritization while 1 means full prioritization
    priority_prob_alpha=0.6,
    # (Float type)  How much correction is used: 0 means no correction while 1 means full correction
    # TODO(pu): test effect of 0.4->1
    priority_prob_beta=0.4,
    prioritized_replay_eps=1e-6,

    root_dirichlet_alpha=0.3,
    root_exploration_fraction=0.25,
    auto_td_steps=int(0.3 * 2e5),
    auto_td_steps_ratio=0.3,

    # UCB formula
    pb_c_base=19652,
    pb_c_init=1.25,

    support_size=300,
    value_support=DiscreteSupport(-300, 300, delta=1),
    reward_support=DiscreteSupport(-300, 300, delta=1),
    max_grad_norm=10,

    test_interval=10000,
    log_interval=1000,
    vis_interval=1000,
    checkpoint_interval=100,
    target_model_interval=200,
    save_ckpt_interval=10000,
    discount=0.997,
    dirichlet_alpha=0.3,
    value_delta_max=0.01,
    num_actors=1,
    # network initialization/ & normalization
    episode_life=True,
    # replay window
    start_transitions=8,
    transition_num=1,
    # frame skip & stack observation
    frame_skip=4,
    frame_stack_num=4,
    # coefficient
    reward_loss_coeff=1,
    value_loss_coeff=0.25,
    policy_loss_coeff=1,
    consistency_coeff=2,
    # reward sum
    lstm_hidden_size=512,
    lstm_horizon_len=5,
    # siamese
    proj_hid=1024,
    proj_out=1024,
    pred_hid=512,
    pred_out=1024,

    bn_mt=0.1,
    blocks=1,  # Number of blocks in the ResNet
    channels=64,  # Number of channels in the ResNet
    reduced_channels_reward=16,  # x36 Number of channels in reward head
    reduced_channels_value=16,  # x36 Number of channels in value head
    reduced_channels_policy=16,  # x36 Number of channels in policy head
    resnet_fc_reward_layers=[32],  # Define the hidden layers in the reward head of the dynamic network
    resnet_fc_value_layers=[32],  # Define the hidden layers in the value head of the prediction network
    resnet_fc_policy_layers=[32],  # Define the hidden layers in the policy head of the prediction network
))

game_config = GameBaseConfig(game_config)
