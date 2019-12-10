# SC2Learner (TStarBot1) - Macro-Action-Based StarCraft-II Reinforcement Learning Environment



*[SC2Learner](https://github.com/Tencent/TStarBot1)* is a *macro-action*-based [StarCraft-II](https://en.wikipedia.org/wiki/StarCraft_II:_Wings_of_Liberty) reinforcement learning research platform.
It exposes the re-designed StarCraft-II action space, which has more than one hundred discrete macro actions, based on the raw APIs exposed by DeepMind and Blizzard's [PySC2](https://github.com/deepmind/pysc2).
The macro action space relieves the learning algorithms from a disastrous burden of directly handling a massive number of atomic keyboard and mouse operations, making learning more tractable.
The environments and wrappers strictly follow the interface of [OpenAI Gym](https://github.com/openai/gym), making it easier to be adapted to many off-the-shelf reinforcement learning algorithms and implementations.

[*TStartBot1*](https://arxiv.org/pdf/1809.07193.pdf), a reinforcement learning agent, is also released with two off-the-shelf reinforcement learning algorithms *Dueling Double Deep Q Network* (DDQN) and *Proximal Policy Optimization* (PPO), as examples.
**Distributed** versions of both algorithms are released, enabling learners to scale up the rollout experience collection across thousands of CPU cores on a cluster of machines.
*TStarBot1* is able to beat **level-9** built-in AI (cheating resources) with **97%** win-rate and **level-10** (cheating insane) with **81%** win-rate.

A whitepaper of *TStarBots* is available at [here](https://arxiv.org/pdf/1809.07193.pdf).

## Table of Contents
- [Installations](#installations)
- [Getting Started](#getting-started)
 	- [Run Random Agent](#run-random-agent)
	- [Train PPO Agent](#train-ppo-agent)
	- [Evaluate PPO Agent](#evaluate-ppo-agent)
	- [Play vs. PPO Agent](#play-vs.-ppo-agent)
	- [Train via Self-play](#train-via-self-play)
- [Environments and Wrappers](environments-and-wrappers)
- [Questions ans Help](questions-and-help)


## Installations

### Prerequisites
- Python >= 3.5 required.
- [PySC2 Extension](https://github.com/Tencent/PySC2TencentExtension) required.

### Setup
Git clone this repository and then install it with
```bash
pip3 install -e sc2learner
```

## Getting Started

### Run Random Agent
Run a random agent playing against a builtin AI of difficulty level 1.
```bash
python3 -m sc2learner.bin.evaluate --agent random --difficulty '1'
```

### Train PPO Agent

To train an agent with PPO algorithm, actor workers and learner worker must be started respectively.
They can run either locally or across separate machines (e.g. actors usually run in a CPU cluster consisting of hundreds of machines with tens of thousands of CPU cores, and a learner runs in a GPU machine).
With the designated ports and learner's IP, rollout trajectories and model parameters are communicated between actors and learner. 
- Start 48 actor workers (run the same script in all actor machines) 
```bash
./experiments/ppo_baseline/actor.sh <partition_name> <num_actor>
```

- Start a learner worker
```bash
./experiments/ppo_baseline/learner.sh <partition_name>
```

Similarly, DQN algorithm can be tried with `sc2learner.bin.train_dqn`.

### Evaluate PPO Agent
After training, the agent's in-game performance can be observed by letting it play the game against a build-in AI of a certain difficulty level.
Win-rate is also estimated meanwhile with multiple such games initialized with different game seeds.
```bash
python3 -m sc2learner.bin.evaluate --agent ppo --difficulty 1 --model_path REPLACE_WITH_YOUR_OWN_MODLE_PATH
```
###

### Play vs. PPO Agent
We can also try ourselves playing against the learned agent by first starting a human player client and then a learned agent.
They can run either locally or remotely.
When run across two machines, `--remote` argument needs to be set for the human player side to create an SSH tunnel to the remote agent's machine and ssh keys must be used for authentication. 

- Start a human player client
```bash
CUDA_VISIBLE_DEVICES= python3 -m pysc2.bin.play_vs_agent --human --map AbyssalReef --user_race zerg
```

- Start a PPO agent
```bash
python3 -m sc2learner.bin.play_vs_ppo_agent --model_path REPLACE_WITH_YOUR_OWN_MODLE_PATH
```

### Train via Self-play

Besides, a self-play training (playing vs. past versions) is also provided to make learning more diversified strategies possible.

- Start Actors
```bash
for i in $(seq 0 48); do
  CUDA_VISIBLE_DEVICES= python3 -m sc2learner.bin.train_ppo_selfplay --job_name=actor --learner_ip localhost &
done;
```

- Start Learner
```bash
CUDA_VISIBLE_DEVICES=0 python3 -m sc2learner.bin.train_ppo_selfplay --job_name learner
```

## Environments and Wrappers

The environments and wrappers strictly follow the interface of [OpenAI Gym](https://github.com/openai/gym).
The macro action space is defined in [`ZergActionWrapper`](https://github.com/Tencent/TStarBot1/blob/dev-open/sc2learner/envs/actions/zerg_action_wrappers.py#L26) and the observation space defined in [`ZergObservationWrapper`](https://github.com/Tencent/TStarBot1/blob/dev-open/sc2learner/envs/observations/zerg_observation_wrappers.py#L24), based on which users can easily make their own changes and restart the training to see what happens.

## Questions and Help
You are welcome to submit questions and bug reports in [Github Issues](https://github.com/Tencent/TStarBot1/issues).
You are also welcome to contribute to this project.
