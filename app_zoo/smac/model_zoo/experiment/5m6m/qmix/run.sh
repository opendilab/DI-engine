srun --mpi=pmi2 --job-name 5m6m_qmix -p VI_SP_Y_V100_A -n1 --gres=gpu:1 --ntasks-per-node=8 python -u train.py
# spring.submit run --mpi=pmi2 --job-name qmix_12_8_3_1 -n 1 --gpu  --cpus-per-task=5 \
# "python -u train.py"
