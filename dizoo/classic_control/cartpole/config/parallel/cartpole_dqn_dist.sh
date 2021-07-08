export PYTHONUNBUFFERED=1
ding -m dist --module config -p slurm -c cartpole_dqn_config.py -s 0 -lh SH-IDC1-10-5-36-161 -clh SH-IDC1-10-5-36-140

srun -p VI_SP_Y_V100_A -w SH-IDC1-10-5-36-161 --gres=gpu:1 ding -m dist --module learner --module-name learner0 -c cartpole_dqn_config.py.pkl -s 0 &
srun -p VI_SP_Y_V100_A -w SH-IDC1-10-5-36-140 ding -m dist --module collector --module-name collector0 -c cartpole_dqn_config.py.pkl -s 0 &
srun -p VI_SP_Y_V100_A -w SH-IDC1-10-5-36-140 ding -m dist --module collector --module-name collector1 -c cartpole_dqn_config.py.pkl -s 0 &

ding -m dist --module coordinator -p slurm -c cartpole_dqn_config.py.pkl -s 0

# the following command is for local test
# ding -m dist --module config -p local -c cartpole_dqn_config.py -s 0
# ding -m dist --module learner --module-name learner0 -c cartpole_dqn_config.py.pkl -s 0 &
# ding -m dist --module collector --module-name collector0 -c cartpole_dqn_config.py.pkl -s 0 &
# ding -m dist --module collector --module-name collector1 -c cartpole_dqn_config.py.pkl -s 0 &
# ding -m dist --module coordinator -p local -c cartpole_dqn_config.py.pkl -s 0
