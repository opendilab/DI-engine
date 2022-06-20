import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--hopper_ours_csv_path', type=str, default='dt_hopper_log/dt_Hopper-v3_log_22-05-20-03-47-07.csv')
parser.add_argument('--hopper_origin_csv_path', type=str, default='dt_hopper_log/dt_Hopper-v3_official.csv')
parser.add_argument('--halfcheetah_ours_csv_path', type=str, default='dt_halfcheetah_log/dt_HalfCheetah-v3_log_22-05-20-06-21-45.csv')
parser.add_argument('--halfcheetah_origin_csv_path', type=str, default='dt_halfcheetah_log/dt_HalfCheetah-v3_official.csv')

parser.add_argument('--save_path', '-sp', type=str, default='./result.png')
args = parser.parse_args()

sns.set_theme(style='darkgrid')
df_hopper_ours = pd.read_csv(args.hopper_ours_csv_path, header=None)
df_hopper_ours.columns = ['time_elapsed', 'total_updates', 'mean_action_loss', 'eval_avg_reward', 'eval_avg_ep_len']
df_hopper_ours.drop(columns=['time_elapsed', 'mean_action_loss', 'eval_avg_ep_len'])

ax = df_hopper_ours.plot(x='total_updates', y='eval_avg_reward', label='hopper_ours')

df_half_ours = pd.read_csv(args.halfcheetah_ours_csv_path, header=None)
df_half_ours.columns = ['time_elapsed', 'total_updates', 'mean_action_loss', 'eval_avg_reward', 'eval_avg_ep_len']
df_half_ours.drop(columns=['time_elapsed', 'mean_action_loss', 'eval_avg_ep_len'])

ax = df_half_ours.plot(x='total_updates', y='eval_avg_reward', label='halfcheetah_ours', ax=ax)


df_half_official = pd.read_csv(args.halfcheetah_origin_csv_path, header=None).iloc[:98]
df_half_official.columns = ['time_elapsed', 'total_updates', 'mean_action_loss', 'eval_avg_reward', 'eval_avg_ep_len', 'd4rl_score']
df_half_official.drop(columns=['time_elapsed', 'mean_action_loss', 'eval_avg_ep_len', 'd4rl_score'])

ax = df_half_official.plot(x='total_updates', y='eval_avg_reward', label='halfcheetah_official', ax=ax)

df_hopper_official = pd.read_csv(args.hopper_origin_csv_path, header=None).iloc[:98]
df_hopper_official.columns = ['time_elapsed', 'total_updates', 'mean_action_loss', 'eval_avg_reward', 'eval_avg_ep_len', 'd4rl_score']
df_hopper_official.drop(columns=['time_elapsed', 'mean_action_loss', 'eval_avg_ep_len', 'd4rl_score'])

ax = df_hopper_official.plot(x='total_updates', y='eval_avg_reward', label='hopper_official', ax=ax)

plt.xlabel('updates')
plt.ylabel('avg reward')
plt.title('avg reward along update times in evaluation')
plt.legend()
plt.savefig(args.save_path)




