import os

slurm_ip = os.environ.get('SLURMD_NODENAME', 'fuck')
print(slurm_ip)