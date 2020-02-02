work_path=$(dirname $0)
job_name=actor_manager
key_word=sc2learner.bin.train_ppo
function launch_func() {
    srun -p $1 -w $2 --gres=gpu:8 --job-name=$job_name python3 -u -m sc2learner.bin.train_ppo --job_name $job_name --config_path $work_path/config.yaml
}
while (true)
do
    content=`squeue -u $3`
    result=$(echo $content | grep $job_name)
    if (["$result" == ""])
    then
        echo launch a new $job_name
        launch_func $1 $2
    fi
    sleep 10s
done
