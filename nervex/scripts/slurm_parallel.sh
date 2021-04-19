export PYTHONUNBUFFERED=1
seed=0
srun -p VI_SP_Y_V100_A --gres=gpu:1 nervex -m parallel -c $1 -s $seed
