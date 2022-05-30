## PYSC2 Env
DI-engine uses standard pysc2 env, you can install it as follow:
```shell
pip install pysc2
```

## SMAC Benchmark

==setting: SC2 version=4.6.2.69232, difficulty=7, 2M env step==


|  3s5z  |  pymarl  |      |DI-engine |          |                             cfg                              |
| :----: | :------: | :--: | :------: | :------: | :----------------------------------------------------------: |
|        | win rate | time | win rate |   time   |                                                              |
|  qmix  |    1     | 9.5h |  **1**   | **3.2h** | dizoo/smac/config/smac_3s5z_qmix_config.py                 |
| collaq |    1     | 28h  |   0.9    | **8.5h** | dizoo/smac/config/smac_3s5z_collaq_config.py               |
|  coma  |    0     | 2.7h | **0.9**  | **2.9h** | dizoo/smac/config/smac_3s5z_coma_config.py                 |
|  qtran |    0.1   | 11.5h | **0.9**  | **4h** | dizoo/smac/config/smac_3s5z_qtran_config.py                 |
|  ippo  |    0.15  |  10.5h  | **0.8**  | **6.8h** |                       |
|  mappo(ours) |    - |  -  | **1**  | **2.4h** |       dizoo/smac/config/smac_3s5z_mappo_config.py            |
|  masac(ours) |    - |  -  | **1**  | **4.4h** |       dizoo/smac/config/smac_3s5z_masac_config.py            |

| 5m_vs_6m |  pymarl  |      |DI-engine |          |                             cfg                              |
| :-------: | :------: | :--: | :------: | :------: | :----------------------------------------------------------: |
|           | win rate | time | win rate |   time   |                                                              |
|   qmix    | **0.76** | 7.5h |   0.6    | **6.5h** | dizoo/smac/config/smac_5m6m_qmix_config.py                 |
|  collaq   |   0.8    | 24h  |   0.7    | **9.5h** | dizoo/smac/config/smac_5m6m_collaq_config.py               |
|   coma    |    0     | 2.5h |    0     |    -     |                                                              |
|  qtran    |    0.7   | 7h   | 0.55  | **5.5h** | dizoo/smac/config/smac_5m6m_qtran_config.py                 |
|  ippo  |      0    |   9.2h   | **0.75**  | **6.9h** |                       |
|  mappo(ours) |      -    |   -   | **0.75**  | **3.2h** |       dizoo/smac/config/smac_5m6m_mappo_config.py            |
|  masac(ours) |      -    |   -   | **1**  | **5.2h** |       dizoo/smac/config/smac_5m6m_masac_config.py            |

|  MMM   |  pymarl  |      |DI-engine |          |                             cfg                              |
| :----: | :------: | :--: | :------: | :------: | :----------------------------------------------------------: |
|        | win rate | time | win rate |   time   |                                                              |
|  qmix  |    1     | 9.5h |  **1**   | **3.5h** | dizoo/smac/config/smac_MMM_qmix_config.py                 |
|  collaq   |  1    | 38h  |   **1**    | **6.7h** | dizoo/smac/config/smac_MMM_collaq_config.py               |
|   coma    |    0.1     | 3h |    **0.9**     |    **2.6h**     |  dizoo/smac/config/smac_MMM_coma_config.py |
|  qtran    |    1   | 8.5h   | **1**  | **5.5h** | dizoo/smac/config/smac_MMM_qtran_config.py                 |
|  ippo  |      0.33    |  7.2h    | **1**  | **4.7h** |                       |
|  mappo(ours) |      -    |    -  | **1**  | **2.7h** |       dizoo/smac/config/smac_MMM_mappo_config.py            |
|  masac(ours) |      -    |   -   | **1**  | **5.2h** |       dizoo/smac/config/smac_MMM_masac_config.py            |


|  MMM2   |  pymarl  |      |DI-engine |          |                             cfg                              |
| :----: | :------: | :--: | :------: | :------: | :----------------------------------------------------------:  |
|        | win rate | time | win rate |   time   |                                                               |
|  qmix  |    0.7   | 10h  |   0.4    | **5.5h** | dizoo/smac/config/smac_MMM2_qmix_config.py                    |
| collaq |    0.9   | 24h  |   0.6    | **13h**  | dizoo/smac/config/smac_MMM2_collaq_config.py                  |
|  coma  |    0     | 3h   |  **0.2** |   3.5h   |                    dizoo/smac/config/smac_MMM2_coma_config.py |
|  qtran |    0     | 8.5h |  0       |   -      |                                                               |
|  ippo  |    0      |  8.3h    | **0.875**  | **6h** |                       |
|  mappo(ours) |    -      |  -    | **1**  | **3.8h** |       dizoo/smac/config/smac_MMM2_mappo_config.py            |
|  masac(ours) |      -    |   -   | **1**  | **7.2h** |       dizoo/smac/config/smac_MMM2_masac_config.py            |


|  3s5z_vs_3s6z   |  MAPPO(Wu)  |      |DI-engine |          |                             cfg                              |
| :----: | :------: | :--: | :------: | :------: | :----------------------------------------------------------:  |
|        | win rate | time | win rate |   time   |                                                               |
|  mappo(ours) |    -      |  -    | **0.88**  | **3.8h** |       dizoo/smac/config/smac_3s5zvs3s6z_mappo_config.py            |
|  masac(ours) |      -    |   -   | **1**  | **7.2h** |       dizoo/smac/config/smac_3s5zvs3s6z_masac_config.py            |

|  8m_vs_9m   |  MAPPO(Wu)  |      |DI-engine |          |                             cfg                              |
| :----: | :------: | :--: | :------: | :------: | :----------------------------------------------------------:  |
|        | win rate | time | win rate |   time   |                                                               |
|  mappo(ours) |    -      |  -    | **1**  | **3.6h** |       dizoo/smac/config/smac_3s5zvs3s6z_mappo_config.py            |
|  masac(ours) |      -    |   -   | **1**  | **6.7h** |       dizoo/smac/config/smac_3s5zvs3s6z_masac_config.py            |

|  10m_vs_11m   |  MAPPO(Wu)  |      |DI-engine |          |                             cfg                              |
| :----: | :------: | :--: | :------: | :------: | :----------------------------------------------------------:  |
|        | win rate | time | win rate |   time   |                                                               |
|  mappo(ours) |    -      |  -    | **1**  | **3.9h** |       dizoo/smac/config/smac_10m11m_mappo_config.py            |
|  masac(ours) |      -    |   -   | **1**  | **6.9h** |       dizoo/smac/config/smac_10m11m_masac_config.py            |


|  25m   |  MAPPO(Wu)  |      |DI-engine |          |                             cfg                              |
| :----: | :------: | :--: | :------: | :------: | :----------------------------------------------------------:  |
|        | win rate | time | win rate |   time   |                                                               |
|  mappo(ours) |    -      |  -    | **1**  | **3.7h** |       dizoo/smac/config/smac_25m_mappo_config.py            |
|  masac(ours) |      -    |   -   | **1**  | **6.4h** |       dizoo/smac/config/smac_25m_masac_config.py            |


|  2c_vs_64zg   |  MAPPO(Wu)  |      |DI-engine |          |                             cfg                              |
| :----: | :------: | :--: | :------: | :------: | :----------------------------------------------------------:  |
|        | win rate | time | win rate |   time   |                                                               |
|  mappo(ours) |    -      |  -    | **1**  | **3.2h** |       dizoo/smac/config/smac_2c64zg_mappo_config.py            |
|  masac(ours) |      -    |   -   | **1**  | **6.1h** |       dizoo/smac/config/smac_2c64zg_masac_config.py            |


|  corridor   |  MAPPO(Wu)  |      |DI-engine |          |                             cfg                              |
| :----: | :------: | :--: | :------: | :------: | :----------------------------------------------------------:  |
|        | win rate | time | win rate |   time   |                                                               |
|  mappo(ours) |    -      |  -    | **1**  | **2.9h** |       dizoo/smac/config/smac_corridor_mappo_config.py            |
|  masac(ours) |      -    |   -   | **1**  | **5.9h** |       dizoo/smac/config/smac_corridor_masac_config.py            |


comment: The time in the table is the time to run 2M env step. 
