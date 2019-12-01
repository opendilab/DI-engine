# Tencent AI Lab PySC2 Extension
**Note: the original Deepmind PySC2 README can be found [here](https://github.com/deepmind/pysc2/blob/master/README.md).**

Besides the "feature_layer" observations/actions interface, 
this Tencent AI Lab fork also exposes the "raw" interface of [`s2client-proto`](https://github.com/Blizzard/s2client-proto) to enable a per-unit-control.

It supports a hybrid use of the two intefaces. For example, consider a two-player game and the code below 
```python
timesteps = env.step(actions)
```
For `player_id = 0`, 
all the `uints` in pb format can be accessed via `timesteps[player_id].observation['units]`,
while the original Deepmind `PySC2` features can still be accessed via `timesteps[player_id].observation['feat_name']`.

For the actions passed in, `acionts[player_id]` can be either a `list` of pb actions or a single Deepmind `PySC2` action. 
(TODO: support a list of hybrid action when necessary).

It goes similar for the other player `player_id = 1`. 

## Installation
git clone the repo, cd to the folder, and run
```bash
pip isntall -e .
```
**Note: the in-place `-e .` installation is REQUIRED,**
as we have binaries (i.e., the `tech_tree` data) shipped with the fork 
and the `-e .` in-place installation makes life easier.

**Note also** that you need pip uninstall the original Deempind PySC2 before installing/using our fork.
Doning so would not be a problem, 
as this fork is compatible with the original Deepmind PySC2.