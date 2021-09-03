<div align="center">
    <a href="http://di-engine.github.io"><img width="500px" height="auto" src="https://github.com/opendilab/DI-engine-docs/blob/main/source/images/di_engine_logo.svg"></a>
</div>

---

[![PyPI](https://img.shields.io/pypi/v/DI-engine)](https://pypi.org/project/DI-engine/)
![Conda](https://anaconda.org/opendilab/di-engine/badges/version.svg)
![Conda update](https://anaconda.org/opendilab/di-engine/badges/latest_release_date.svg)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/DI-engine)
![PyTorch Version](https://img.shields.io/badge/dynamic/json?color=blue&label=pytorch&query=%24.pytorchVersion&url=https%3A%2F%2Fgist.githubusercontent.com%2FPaParaZz1%2F54c5c44eeb94734e276b2ed5770eba8d%2Fraw%2F01437f709b3f2f2fde4abf4d063dc7409066dd50%2Fbadges.json)
![Libraries.io dependency status for GitHub repo](https://img.shields.io/librariesio/github/opendilab/DI-engine)


![Loc](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/HansBug/3690cccd811e4c5f771075c2f785c7bb/raw/loc.json)
![Comments](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/HansBug/3690cccd811e4c5f771075c2f785c7bb/raw/comments.json)


![Style](https://github.com/opendilab/DI-engine/actions/workflows/style.yml/badge.svg)
![Docs](https://github.com/opendilab/DI-engine/actions/workflows/doc.yml/badge.svg)
![Unittest](https://github.com/opendilab/DI-engine/actions/workflows/unit_test.yml/badge.svg)
![Algotest](https://github.com/opendilab/DI-engine/actions/workflows/algo_test.yml/badge.svg)
![Platformtest](https://github.com/opendilab/DI-engine/actions/workflows/platform_test.yml/badge.svg)
[![codecov](https://codecov.io/gh/opendilab/DI-engine/branch/main/graph/badge.svg?token=B0Q15JI301)](https://codecov.io/gh/opendilab/DI-engine)



![GitHub Org's stars](https://img.shields.io/github/stars/opendilab)
[![GitHub stars](https://img.shields.io/github/stars/opendilab/DI-engine)](https://github.com/opendilab/DI-engine/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/opendilab/DI-engine)](https://github.com/opendilab/DI-engine/network)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/opendilab/DI-engine)
[![GitHub issues](https://img.shields.io/github/issues/opendilab/DI-engine)](https://github.com/opendilab/DI-engine/issues)
[![GitHub pulls](https://img.shields.io/github/issues-pr/opendilab/DI-engine)](https://github.com/opendilab/DI-engine/pulls)
[![Contributors](https://img.shields.io/github/contributors/opendilab/DI-engine)](https://github.com/opendilab/DI-engine/graphs/contributors)
[![GitHub license](https://img.shields.io/github/license/opendilab/DI-engine)](https://github.com/opendilab/DI-engine/blob/master/LICENSE)

Updated on 2021.08.03 DI-engine-v0.1.1 (beta)


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

### Other
- [DI-engine-docs](https://github.com/opendilab/DI-engine-docs)
- [treevalue](https://github.com/opendilab/treevalue) (preview)

## Installation

You can simply install DI-engine from PyPI with the following command:
```bash
pip install DI-engine
```

If you use Anaconda or Miniconda, you can install DI-engine from conda-forge through the following command:
```bash
conda install -c opendilab di-engine
```

For more information about installation, you can refer to [installation](https://opendilab.github.io/DI-engine/installation/index.html).

## Documentation

The detailed documentation are hosted on [doc](https://opendilab.github.io/DI-engine/)([中文文档](https://di-engine-docs.readthedocs.io/en/main-zh/)).

## Quick Start

[3 Minutes Kickoff](https://opendilab.github.io/DI-engine/quick_start/index.html)

[3 Minutes Kickoff(colab)](https://colab.research.google.com/drive/1J29voOD2v9_FXjW-EyTVfRxY_Op_ygef#scrollTo=MIaKQqaZCpGz)

[3分钟上手中文版(kaggle)](https://www.kaggle.com/shenzhenperson/di-engine)

**Bonus: Train RL agent in one line code:**
```bash
ding -m serial -e cartpole -p dqn -s 0
```

## Contributing
We appreciate all contributions to improve DI-engine, both algorithms and system designs. Please refer to CONTRIBUTING.md for more guides. And our roadmap can be accessed by [this link](https://github.com/opendilab/DI-engine/projects/1).

And users can join our [slack communication channel](https://join.slack.com/t/opendilab/shared_invite/zt-v9tmv4fp-nUBAQEH1_Kuyu_q4plBssQ) for more detailed discussion.


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
