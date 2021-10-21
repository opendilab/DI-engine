## PYSC2 Env
DI-engine use modified pysc2 env (for more maps and agent vs agent training), and you need to install [DI-star](https://github.com/opendilab/DI-star#installation) first, then you can import related package like this:
```python
import ctools.pysc2
from ctools.pysc2.maps import lib
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
|  mappo |    0.936 |  10.5h  | **1**  | **7.4h** |       dizoo/smac/config/smac_3s5z_mappo_config.py            |


| 5m vs. 6m |  pymarl  |      |DI-engine |          |                             cfg                              |
| :-------: | :------: | :--: | :------: | :------: | :----------------------------------------------------------: |
|           | win rate | time | win rate |   time   |                                                              |
|   qmix    | **0.76** | 7.5h |   0.6    | **6.5h** | dizoo/smac/config/smac_5m6m_qmix_config.py                 |
|  collaq   |   0.8    | 24h  |   0.7    | **9.5h** | dizoo/smac/config/smac_5m6m_collaq_config.py               |
|   coma    |    0     | 2.5h |    0     |    -     |                                                              |
|  qtran    |    0.7   | 7h   | 0.55  | **5.5h** | dizoo/smac/config/smac_5m6m_qtran_config.py                 |
|  ippo  |      0    |   9.2h   | **0.75**  | **6.9h** |                       |
|  mappo |      0    |   9.4h   | **0.75**  | **7.2h** |       dizoo/smac/config/smac_5m6m_mappo_config.py            |

|  MMM   |  pymarl  |      |DI-engine |          |                             cfg                              |
| :----: | :------: | :--: | :------: | :------: | :----------------------------------------------------------: |
|        | win rate | time | win rate |   time   |                                                              |
|  qmix  |    1     | 9.5h |  **1**   | **3.5h** | dizoo/smac/config/smac_MMM_qmix_config.py                 |
|  collaq   |  1    | 38h  |   **1**    | **6.7h** | dizoo/smac/config/smac_MMM_collaq_config.py               |
|   coma    |    0.1     | 3h |    **0.9**     |    **2.6h**     |  dizoo/smac/config/smac_MMM_coma_config.py |
|  qtran    |    1   | 8.5h   | **1**  | **5.5h** | dizoo/smac/config/smac_MMM_qtran_config.py                 |
|  ippo  |      0.33    |  7.2h    | **1**  | **4.7h** |                       |
|  mappo |      1    |    7.2h  | **1**  | **3.7h** |       dizoo/smac/config/smac_MMM_mappo_config.py            |

|  MMM2   |  pymarl  |      |DI-engine |          |                             cfg                              |
| :----: | :------: | :--: | :------: | :------: | :----------------------------------------------------------:  |
|        | win rate | time | win rate |   time   |                                                               |
|  qmix  |    0.7   | 10h  |   0.4    | **5.5h** | dizoo/smac/config/smac_MMM2_qmix_config.py                    |
| collaq |    0.9   | 24h  |   0.6    | **13h**  | dizoo/smac/config/smac_MMM2_collaq_config.py                  |
|  coma  |    0     | 3h   |  **0.2** |   3.5h   |                    dizoo/smac/config/smac_MMM2_coma_config.py |
|  qtran |    0     | 8.5h |  0       |   -      |                                                               |
|  ippo  |    0      |  8.3h    | **0.875**  | **6h** |                       |
|  mappo |    0      |  8.3h    | **1**  | **9h** |       dizoo/smac/config/smac_MMM2_mappo_config.py            |

comment: The time in the table is the time to run 2M env step. The IPPO and MAPPO were experimented with DI-engine and the epymarl open source library. The epymarl library commit id is ee85dd0.
