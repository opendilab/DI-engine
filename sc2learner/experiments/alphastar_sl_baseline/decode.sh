partition=$1
workstation=$2
replays=$3
outputdir=$4
logpath=$5

nohup srun -p $partition -w $workstation --job-name=cpu python -u -m sc2learner.bin.replay_decode \
    --parallel 1 \
    --step_mul 1 \
    --replays $replays \
    --output_dir $outputdir \
> $logpath 2>&1 &

