# Usage: viz.sh <port_no>
tensorboard --logdir=experiments --port=$1 --samples_per_plugin "images=100"
