partition=$1
workstation=$2
replays=$3
outputdir=$4
process_num=$5
logpath=$6

nohup srun -p $partition -w $workstation --job-name=cpu python -u -m sc2learner.data.offline.replay_decode_game_loop \
    --replays $replays \
    --output_dir $outputdir \
    --version 4.9.3 \
    --process_num $process_num \
> $logpath 2>&1 &

# srun -p VI_SP_VA_V100 -w SH-IDC1-10-5-36-86 --job-name=cpua python -u -m sc2learner.bin.replay_decode_game_loop --replays /mnt/lustre/zhangming/data/listtemp410/VI_SP_Y_V100_B/SH-IDC1-10-5-36-155 --output_dir /mnt/lustre/zhangming/data/Replays_decode_valid_410 --parallel 1
