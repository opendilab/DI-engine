## BitFlip Environment
A simple environment to flip a 01 sequence into a specific state. With the bits number increasing, the task becomes harder. 
Well suited for testing Hindsight Experience Replay.

## DI-engine's HER on BitFlip

The table shows how many envsteps are needed at least to converge for PureDQN and HER-DQN implemented in DI-engine. '-' means no convergence in 20M envsteps.

| n_bit  | PureDQN | HER-DQN |
| ------ | ------- | ------- |
| 15     | -       | 150K    |
| 20     | -       | 1.5M    |
DI-engine's HER-DQN can converge 

You can refer to the RL algorithm doc for implementation and experiment details.
