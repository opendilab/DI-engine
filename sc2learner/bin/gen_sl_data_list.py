from sc2learner.dataset import get_replay_list

home_race = None
away_race = None
min_mmr = 1000
replay_dir = '/mnt/lustre/niuyazhe/data/sl_data'
output_path = 'test_sl.txt'
get_replay_list(replay_dir, output_path, min_mmr=min_mmr,
                home_race=home_race, away_race=away_race)
