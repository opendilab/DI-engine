<div align="center">
    <a href="http://di-engine.github.io"><img width="500px" height="auto" src="ding/docs/source/images/di_engine_logo.svg"></a>
</div>

---

[![PyPI](https://img.shields.io/pypi/v/DI-engine)](https://pypi.org/project/DI-engine/)
![Style](https://github.com/opendilab/DI-engine/actions/workflows/style.yml/badge.svg)
![Docs](https://github.com/opendilab/DI-engine/actions/workflows/doc.yml/badge.svg)
![Unittest](https://github.com/opendilab/DI-engine/actions/workflows/unit_test.yml/badge.svg)
![Algotest](https://github.com/opendilab/DI-engine/actions/workflows/algo_test.yml/badge.svg)
![Platformtest](https://github.com/opendilab/DI-engine/actions/workflows/platform_test.yml/badge.svg)


[![GitHub issues](https://img.shields.io/github/issues/opendilab/DI-engine)](https://github.com/opendilab/DI-engine/issues)
[![GitHub stars](https://img.shields.io/github/stars/opendilab/DI-engine)](https://github.com/opendilab/DI-engine/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/opendilab/DI-engine)](https://github.com/opendilab/DI-engine/network)
[![GitHub license](https://img.shields.io/github/license/opendilab/DI-engine)](https://github.com/opendilab/DI-engine/blob/master/LICENSE)

Updated on 2021.07.08 DI-engine-v0.1.0 (beta)


## Introduction to DI-engine (beta)
DI-engine is a generalized Decision Intelligence engine. It supports most basic deep reinforcement learning (DRL) algorithms, such as DQN, PPO, SAC, and domain-specific algorithms like QMIX in multi-agent RL, GAIL in 
inverse RL, and RND in exploration problems. Various training pipelines and customized decision AI applications are also supported. Have fun with exploration and exploitation.

### Application
- [DI-star](https://github.com/opendilab/DI-star)
- [DI-drive](https://github.com/opendilab/DI-drive)

### System Optimization and Design
- [DI-orchestrator](https://github.com/opendilab/DI-orchestrator)
- [DI-hpc](https://github.com/opendilab/DI-hpc)
- [DI-store](https://github.com/opendilab/DI-store)


## Installation

You can simply install DI-engine from PyPI with the following command:
```bash
pip install DI-engine
```

If you use Anaconda or Miniconda, you can install DI-engine from conda-forge through the following command:
```bash
conda -c conda-forge install DI-engine
```

For more information about installation, you can refer to [installation](https://opendilab.github.io/DI-engine/installation/index.html).

## Documentation

The detailed documentation are hosted on [doc](https://opendilab.github.io/DI-engine/).

## Quick Start

[3 Minutes Kickoff](https://opendilab.github.io/DI-engine/quick_start/index.html)

Bonus: Train RL agent in one line code:
```bash
ding -m serial -e cartpole -p dqn -s 0
```

## Contributing
We appreciate all contributions to improve DI-engine, both algorithms and system designs. Please refer to CONTRIBUTING.md for more guides.

## Citation
```latex
@misc{ding,
    title={{DI-engine: OpenDILab} Decision Intelligence Engine},
    author={DI-engine Contributors},
    publisher = {GitHub},
    howpublished = {\url{https://github.com/opendilab/DI-engine}},
    year={2021},
}
```

## License
DI-engine released under the Apache 2.0 license.
