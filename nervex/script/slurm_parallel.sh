seed=0
nervex -m parallel -p slurm -c $1 -s $seed --actor_host $2 --learner_host $3
