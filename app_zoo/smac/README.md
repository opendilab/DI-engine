## SMAC Benchmark

==setting: SC2 version=4.6.2.69232, difficulty=7==

目前表格中DI-engine的结果是tensorboard的smooth=0.7时reward的最大值

|  3s5z  |  pymarl  |      |DI-engine |          |                             cfg                              |
| :----: | :------: | :--: | :------: | :------: | :----------------------------------------------------------: |
|        | win rate | time | win rate |   time   |                                                              |
|  qmix  |    1     | 9.5h |  **1**   | **3.2h** | app_zoo/smac/config/smac_3s5z_qmix_config.py                 |
| collaq |    1     | 28h  |   0.9    | **8.5h** | app_zoo/smac/config/smac_3s5z_collaq_config.py               |
|  coma  |    0     | 2.7h | **0.9**  | **2.9h** | app_zoo/smac/config/smac_3s5z_coma_config.py                 |


| 5m vs. 6m |  pymarl  |      |DI-engine |          |                             cfg                              |
| :-------: | :------: | :--: | :------: | :------: | :----------------------------------------------------------: |
|           | win rate | time | win rate |   time   |                                                              |
|   qmix    | **0.76** | 7.5h |   0.6    | **6.5h** | app_zoo/smac/config/smac_5m6m_qmix_config.py                 |
|  collaq   |   0.8    | 24h  |   0.7    | **9.5h** | app_zoo/smac/config/smac_5m6m_collaq_config.py               |
|   coma    |    0     | 2.5h |    0     |    -     |                                                              |

备注  2M env step, time是跑2M env step的时间, 测速是一台V100 single GPU
