from gym.envs.registration import register

register(id='MiniGrid-AKTDT-v0', entry_point='dizoo.minigrid.envs:AppleKeyToDoorTreasure')

register(id='MiniGrid-AKTDT-13x13-v0', entry_point='dizoo.minigrid.envs:AppleKeyToDoorTreasure_13x13')

register(id='MiniGrid-AKTDT-19x19-v0', entry_point='dizoo.minigrid.envs:AppleKeyToDoorTreasure_19x19')
