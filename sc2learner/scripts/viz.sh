# Usage: viz.sh <port_no>
tensorboard --logdir=$1 --port=$2 --samples_per_plugin "images=100"
