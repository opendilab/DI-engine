## SMAC Benchmark

==setting: SC2 version=4.6.2.69232, difficulty=7==

目前表格中nervex的结果是tensorboard的smooth=0.7时reward的最大值

#### 3s5z

| 算法        | pymarl |      | nerveX |      | nervex config                                                |
| ----------- | ------ | ---- | ------ | ---- | ------------------------------------------------------------ |
| env step    | 2M     | 3M   | 2M     | 3M   |                                                              |
| vdn         |        |      |        |      |                                                              |
| qmix        |        |      | 0.53   | 0.66 | [cfg](https://gitlab.bj.sensetime.com/open-XLab/cell/nerveX/tree/lj-smac-dev/app_zoo/smac/optimal_config/smac_3s5z_qmix_config5.py) |
| collaq      |        |      | 0.34   | 0.34 | [cfg](https://gitlab.bj.sensetime.com/open-XLab/cell/nerveX/tree/lj-smac-dev/app_zoo/smac/optimal_config/smac_3s5z_collaQ_config.py) |
| coma        |        |      |        |      | [cfg](https://gitlab.bj.sensetime.com/open-XLab/cell/nerveX/tree/lj-smac-dev/app_zoo/smac/optimal_config/smac_3s5z_coma_config4.py) |
| qtrans      |        |      |        |      |                                                              |
| wqmix(sota) |        |      |        |      |                                                              |



#### 5m vs. 6m

| 算法        | pymarl |      | nerveX |      | nervex config                                                |
| ----------- | ------ | ---- | ------ | ---- | ------------------------------------------------------------ |
| env step    | 2M     | 3M   | 2M     | 3M   |                                                              |
| vdn         |        |      |        |      |                                                              |
| qmix        |        |      | 0.5    | 0.67 | [cfg](https://gitlab.bj.sensetime.com/open-XLab/cell/nerveX/tree/lj-smac-dev/app_zoo/smac/optimal_config/smac_5m6m_qmix_config31_2.py) |
| collaq      |        |      | 0.42   | 0.59 | [cfg](https://gitlab.bj.sensetime.com/open-XLab/cell/nerveX/tree/lj-smac-dev/app_zoo/smac/optimal_config/smac_5m6m_collaQ_config1.py) |
| coma        |        |      |        |      |                                                              |
| qtrans      |        |      |        |      |                                                              |
| wqmix(sota) |        |      |        |      |                                                              |