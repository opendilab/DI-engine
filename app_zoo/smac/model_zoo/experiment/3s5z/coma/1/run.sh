srun --mpi=pmi2 --job-name 3s5z_coma -p VI_SP_Y_V100_A -n1 --gres=gpu:1 -w SH-IDC1-10-5-36-152 --ntasks-per-node=56 python -u train.py
# spring.submit run --mpi=pmi2 --job-name qmix_12_8_3_1 -n 1 --gpu  --cpus-per-task=5 \
# "python -u train.py"
