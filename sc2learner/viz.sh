# Usage: viz.sh <port_no>
python ~/.local/lib/python3.6/site-packages/tensorboard/main.py --logdir=$PWD/experiments/ --port=$1 --samples_per_plugin "images=100"
