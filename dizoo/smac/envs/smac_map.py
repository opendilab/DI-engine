from ctools.pysc2.maps import lib
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
    "8m": {
        "n_agents": 8,
        "n_enemies": 8,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "25m": {
        "n_agents": 25,
        "n_enemies": 25,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "5m_vs_6m": {
        "n_agents": 5,
        "n_enemies": 6,
        "limit": 70,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "8m_vs_9m": {
        "n_agents": 8,
        "n_enemies": 9,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "10m_vs_11m": {
        "n_agents": 10,
        "n_enemies": 11,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "27m_vs_30m": {
        "n_agents": 27,
        "n_enemies": 30,
        "limit": 180,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "MMM": {
        "n_agents": 10,
        "n_enemies": 10,
        "limit": 150,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "MMM2": {
        "n_agents": 10,
        "n_enemies": 12,
        "limit": 180,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 3,
        "map_type": "MMM",
    },
    "2s3z": {
        "n_agents": 5,
        "n_enemies": 5,
        "limit": 120,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
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
    "infestor_viper": {
        "n_agents": 2,
        "n_enemies": 9,
        "limit": 150,
        "a_race": "Z",
        "b_race": "Z",
        "unit_type_bits": 2,
        "map_type": "infestor_viper"
    },
    "3s5z_vs_3s6z": {
        "n_agents": 8,
        "n_enemies": 9,
        "limit": 170,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 2,
        "map_type": "stalkers_and_zealots",
    },
    "3s_vs_3z": {
        "n_agents": 3,
        "n_enemies": 3,
        "limit": 150,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 0,
        "map_type": "stalkers",
    },
    "3s_vs_4z": {
        "n_agents": 3,
        "n_enemies": 4,
        "limit": 200,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 0,
        "map_type": "stalkers",
    },
    "3s_vs_5z": {
        "n_agents": 3,
        "n_enemies": 5,
        "limit": 250,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 0,
        "map_type": "stalkers",
    },
    "1c3s5z": {
        "n_agents": 9,
        "n_enemies": 9,
        "limit": 180,
        "a_race": "P",
        "b_race": "P",
        "unit_type_bits": 3,
        "map_type": "colossi_stalkers_zealots",
    },
    "2m_vs_1z": {
        "n_agents": 2,
        "n_enemies": 1,
        "limit": 150,
        "a_race": "T",
        "b_race": "P",
        "unit_type_bits": 0,
        "map_type": "marines",
    },
    "corridor": {
        "n_agents": 6,
        "n_enemies": 24,
        "limit": 400,
        "a_race": "P",
        "b_race": "Z",
        "unit_type_bits": 0,
        "map_type": "zealots",
    },
    "6h_vs_8z": {
        "n_agents": 6,
        "n_enemies": 8,
        "limit": 150,
        "a_race": "Z",
        "b_race": "P",
        "unit_type_bits": 0,
        "map_type": "hydralisks",
    },
    "2s_vs_1sc": {
        "n_agents": 2,
        "n_enemies": 1,
        "limit": 300,
        "a_race": "P",
        "b_race": "Z",
        "unit_type_bits": 0,
        "map_type": "stalkers",
    },
    "so_many_baneling": {
        "n_agents": 7,
        "n_enemies": 32,
        "limit": 100,
        "a_race": "P",
        "b_race": "Z",
        "unit_type_bits": 0,
        "map_type": "zealots",
    },
    "bane_vs_bane": {
        "n_agents": 24,
        "n_enemies": 24,
        "limit": 200,
        "a_race": "Z",
        "b_race": "Z",
        "unit_type_bits": 2,
        "map_type": "bane",
    },
    "2c_vs_64zg": {
        "n_agents": 2,
        "n_enemies": 64,
        "limit": 400,
        "a_race": "P",
        "b_race": "Z",
        "unit_type_bits": 0,
        "map_type": "colossus",
    },
}

for name in map_param_registry.keys():
    globals()[name] = type(name, (SMACMap, ), dict(filename=name))


def get_map_params(map_name):
    return map_param_registry[map_name]
