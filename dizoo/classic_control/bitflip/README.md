## Bitflip Environment
A simple environment to flip coin into specific state. With the number of coin increasing, the task become harder. Well suited for testing Hindsight Experience Replay.

## Ding on Bitflip

The following table shows number of episodes for pure DQN/HER to converage on Bitflip environment. '-' means not converage in 2M episode.

| n_FLIP | dqn   | her    |
| ------ | ----- | ------ |
| 8      | 25688 | 6456   |
| 10     | 181896 | 26832  |
| 15     | -     | 360464 |
| 20     | -     | 1965320 |

