# Usage: viz.sh <experiment_foulder_name> <port_no>
python ~/.local/lib/python3.6/site-packages/tensorboard/main.py --logdir=$PWD/experiments/$1 --port=$2 --samples_per_plugin "images=100"
