![system](http://gitlab.bj.sensetime.com/xialiqiao/SenseStar/-/wikis/uploads/a586d7ac741f472e11b14c04271f187e/Screen_Shot_2020-04-10_at_12.42.01.png)
# Refatoring Branch Quickstart Manual 

Updated on 2020.04.10

Documents: http://xialiqiao.pages.gitlab.bj.sensetime.com/SenseStar 

## Brief Intro on Config System

* **config是个多层级的字典**：最外层为evaluate、common、model、train等大域，内部各自有小域或项
* **只有一个config会被使用**：用户输入的config与各层级默认config融合之后的config会被使用
* **所有缺失值将由默认config填充**：无论是model这种大域，或是其中任何一个子域，系统已有的默认config文件如下：
  1. `sc2learner/agent/model/alphastar/actor_critic_default_config.yaml` - 填充model大域
  2. `sc2learner/worker/learner/alphastar_sl_learner_default_config.yaml` - 填充learner有关域
  3. `sc2learner/train/train_default_config.yaml` - 填充common和logger大域
* 默认config文件结构与完整config相同，即也是从大域开始逐步填充小域和项（见`actor_critic_default_config.yaml``），默认config文件内有丰富的注释 (TODO)
* 在SL实验中，config系统会在实验开始前**自动存储**目前使用的config于`data/experiment_config.yaml`文件中。该文件直接兼容config读取模块，即可直接作为下一次实验的config传入`python xxx.py --config_path data/experiment_config.yaml`
* 本文末展示了一个典型的config

## Install on Each Lustre Instance
```bash
source r0.3.0
git clone <the repo>
cd SenseStar
pip install -e . --user
```

## Supervised Learner

1. Test with fake dataset

```bash
cd sc2learner
python train/train_sl.py --use_fake_dataset --config_path tests/test_alphastar_sl_config.yaml
```

2. Test without fake dataset

```bash
cd sc2learner
python train/train_sl.py --nouse_fake_dataset --config_path tests/test_alphastar_sl_config.yaml
```

3. Supervised Learning of AlphaStart (formal experiment)

```bash
# Run the code in cluster management node!
# The x_cerebra is the name of the partition you want to run on
cd sc2learner/experiments/alphastar_sl_baseline
bash train_multi.sh x_cerebra
```



## Reinforcement Learning Actor

1. Test the actor

```bash
cd sc2learner/tests
python test_alphastar_actor.py --config_path test_alphastar_actor.yaml
```



## Setup Tensorboard

```bash
# The "experiments" is the logdir of tensorboard, 10045 is the port
# You can then run 10.198.6.31:10045 in your browser
cd sc2learner
bash scripts/viz.sh experiments 10045
```

## Evaluation of a checkpoint (or comparing two checkpoints)

Assumming `eval.yaml` is properly edited and in the same folder of `eval.sh`

```bash
cd sc2learner/experiments/alphastar_xx
# Running on partition cpu (cpu only), assumming use_cuda in eval.yaml is set to False
bash eval.sh cpu

# Running on x_cerebra with GPU
bash eval.sh x_cerebra --gres=gpu:1
```

## Example Config for Evaluation

```yaml
evaluate:
    fix_seed: False  # if set to False, seed for games will be set as range(0, num_episodes)
    seed: 0
    use_multiprocessing: False  # multiprocessing may be incompatible with GPU inference
    num_episodes: 4  # run how many games for statistics?
    num_instance_per_node: 2
    game_type: 'self_play'  # self_play or game_vs_bot
    map_name: 'KairosJunction'
    # see https://github.com/deepmind/pysc2/blob/master/pysc2/env/sc2_env.py for all agent property options
    home_race: 'zerg'  # for agent0
    away_race: 'zerg'  # for bot or agent1
    bot_difficulty: 'very_easy'
    bot_build: 'random'
    save_replay: False
    replay_path: 'test'  # find recorded replays in ~/StarCraftII/Replays/test (or $SC2PATH/Replays/test)
    # for game_vs_bot, only paths for agent0 are used
    stat_path:
        agent0: '/mnt/lustre/share_data/niuyazhe/Zerg_Zerg_6280_d35c22b2d7e462f1481621cbf765709961e3f9a2a99f8f6c6fa814ccffc831d6.stat_processed'
        agent1: '/mnt/lustre/share_data/niuyazhe/Zerg_Zerg_6280_d35c22b2d7e462f1481621cbf765709961e3f9a2a99f8f6c6fa814ccffc831d6.stat_processed'
    model_path:
        agent0: '/mnt/lustre/share_data/niuyazhe/iterations_2200.pth.tar'
        agent1: '/mnt/lustre/share_data/niuyazhe/iterations_2200.pth.tar'
    show_system_stat: False  # print GPU memory usage and timing (WIP)
train:
    use_cuda: False
env:
    screen_resolution: [128, 128]
    default_step_mul: 8  # not used
    game_steps_per_episode: 600  # game length cutoff
    disable_fog: False
    realtime: False  # realtime mode, will introduce (unpredictable) env delay, see https://github.com/deepmind/pysc2/blob/master/pysc2/env/remote_sc2_env.py ,incompatible with action_delays
    use_stat: True
    beginning_build_order_num: 20
    use_global_cumulative_stat: True
    use_available_action_transform: True
    temperature: 1.0
    crop_map_to_playable_area: False
    action_delays: [0.5, 0.25, 0.25]  # list of probablities of delay from 1 to n or null for delay=1
model:
    action_type: fixed
```

## Example of Supervised Learner Config

```yaml
common:
  load_path: ''
  name: ZergStat
  only_evaluate: false
  save_path: tests
  time_wrapper_type: cuda
data:
  eval:
    batch_size: 1
    beginning_build_order_num: 20
    dataloader_type: epoch
    dataset_type: fake
    replay_list: null
    use_available_action_transform: true
    use_ceph: false
    use_distributed: false
    use_global_cumulative_stat: false
    use_stat: true
  train:
    batch_size: 3
    beginning_build_order_num: 20
    beginning_build_order_prob: 0.8
    cumulative_stat_prob: 0.5
    dataloader_type: iter
    dataset_type: fake
    replay_list: null
    slide_window_step: 1
    trajectory_len: 5
    trajectory_type: sequential
    use_available_action_transform: true
    use_ceph: false
    use_distributed: false
    use_global_cumulative_stat: false
    use_stat: true
env: {}
logger:
  eval_freq: 200
  print_freq: 1
  save_freq: 200
  var_record_type: alphastar
model:
  encoder:
    core_lstm:
      hidden_size: 384
      input_size: 1792
      num_layers: 3
    obs_encoder:
      encoder_names:
      - scalar_encoder
      - spatial_encoder
      - entity_encoder
      entity_encoder:
        activation: relu
        dropout_ratio: 0.1
        head_dim: 128
        head_num: 2
        hidden_dim: 1024
        input_dim: 2102
        layer_num: 3
        mlp_num: 2
        output_dim: 256
      scalar_encoder:
        activation: relu
        begin_num: 20
        output_dim: 1280
        use_stat: true
      spatial_encoder:
        activation: relu
        down_channels:
        - 64
        - 128
        - 128
        downsample_type: conv2d
        fc_dim: 256
        input_dim: 52
        norm_type: BN
        project_dim: 32
        resblock_num: 4
    scatter:
      input_dim: 256
      output_dim: 32
  freeze_targets: []
  policy:
    head:
      action_type_head:
        action_map_dim: 256
        action_num: 327
        activation: relu
        context_dim: 256
        gate_dim: 1024
        input_dim: 384
        norm_type: LN
        res_dim: 256
        res_num: 16
      delay_head:
        activation: relu
        decode_dim: 256
        delay_dim: 128
        delay_encode_dim: 6
        delay_map_dim: 256
        input_dim: 1024
      head_names:
      - action_type_head
      - delay_head
      - queued_head
      - selected_units_head
      - target_unit_head
      - location_head
      location_head:
        activation: relu
        map_skip_dim: 128
        output_type: cls
        res_dim: 128
        res_num: 4
        reshape_channel: 4
        reshape_size:
        - 16
        - 16
        upsample_dims:
        - 128
        - 64
        - 16
        - 1
        upsample_type: deconv
      queued_head:
        activation: relu
        decode_dim: 256
        input_dim: 1024
        queued_dim: 2
        queued_map_dim: 256
      selected_units_head:
        activation: relu
        entity_embedding_dim: 256
        func_dim: 256
        hidden_dim: 32
        input_dim: 1024
        key_dim: 32
        max_entity_num: 64
        num_layers: 1
        unit_type_dim: 259
      target_unit_head:
        activation: relu
        entity_embedding_dim: 256
        func_dim: 256
        input_dim: 1024
        key_dim: 32
        unit_type_dim: 259
    location_expand_ratio: 2
  state_dict_mask: []
  use_value_network: false
train:
  criterion:
    kwargs:
      smooth_ratio: 0.1
    type: cross_entropy
  learning_rate: 1e-3
  max_epochs: 1000
  temperature: 1.0
  use_cuda: true
  use_distributed: false
  weight_decay: 1e-5

```