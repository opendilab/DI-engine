'''
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. filter replays with specific constraints and generate 
    the filterd replay list
'''
from sc2learner.dataset import get_replay_list

home_race = None
away_race = None
# ISSUE(zh) replays whose mmr below 1000 have already been ditched in replay_decode.py, min_mmr should be above 1000
min_mmr = 1000
replay_dir = '/mnt/lustre/niuyazhe/data/sl_data'
output_path = 'test_sl.txt'
get_replay_list(replay_dir, output_path, min_mmr=min_mmr,
                home_race=home_race, away_race=away_race)
