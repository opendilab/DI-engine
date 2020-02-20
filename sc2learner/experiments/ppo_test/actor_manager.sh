work_path=$(dirname $0)

if [ "$#" -ne 1 ]; then
    job_name=actor_manager
elif [ "$1" = "data" ]; then
    job_name=actor_data_manager
elif [ "$1" = "model" ]; then
    job_name=actor_model_manager
fi 

key_word=sc2learner.bin.train_ppo
function launch_func() {
    python3 -u -m sc2learner.bin.train_ppo --job_name $job_name --config_path $work_path/config.yaml
}
while (true)
do
    content=`ps -u | grep $key_word`
    result=$(echo $content | grep $job_name)
    if (["$result" == ""])
    then
        echo launch a new $job_name
        launch_func
    fi
    sleep 10s
done
