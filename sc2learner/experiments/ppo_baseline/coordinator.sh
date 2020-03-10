work_path=$(dirname $0)
job_name=coordinator
key_word=sc2learner.bin.train_ppo
function launch_func {
    if ([ "$1" = "" ])
    then
        load_arg=""
    else
        load_arg=" --load_path $work_path/checkpoints/coordinator_iter$1.pickle"
    fi
    python3 -u -m sc2learner.bin.train_ppo --job_name $job_name --config_path $work_path/config.yaml$load_arg
}
while (true)
do
    content=`ps -u | grep $key_word`
    result=$(echo $content | grep $job_name)
    if ([ "$result" = "" ])
    then
        echo launch a new $job_name
        launch_func $1
    fi
    sleep 10s
done
