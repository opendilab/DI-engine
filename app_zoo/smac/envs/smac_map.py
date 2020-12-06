from pysc2.maps import lib
import os


class SMACMap(lib.Map):
    directory = os.path.join(os.path.dirname(__file__), "maps/SMAC_Maps")
    download = "https://github.com/oxwhirl/smac#smac-maps"
    players = 2
    step_mul = 8
    game_steps_per_episode = 0


# Copied from smac/env/starcraft2/maps/smac_maps.py
map_param_registry = {
    "3m": {
        "n_agents": 3,
        "n_enemies": 3,
        "limit": 60,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "3s5z": {
        "n_agents": 8,
        "n_enemies": 8,
        "limit": 150,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
}

for name in map_param_registry.keys():
    globals()[name] = type(name, (SMACMap, ), dict(filename=name))


def get_map_params(map_name):
    return map_param_registry[map_name]
