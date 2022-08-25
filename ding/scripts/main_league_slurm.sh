#!/usr/bin/env bash

export LC_ALL=en_US.utf-8
export LANG=en_US.utf-8
BASEDIR=$(dirname "$0")
# srun -p partition_name --quotatype=reserved --mpi=pmi2 -n6 --ntasks-per-node=3 bash ding/scripts/main_league_slurm.sh
ditask --package $BASEDIR/../entry --main main_league.main --platform slurm --platform-spec '{"tasks":[{"labels":"league,collect","node_ids":10},{"labels":"league,collect","node_ids":11},{"labels":"evaluate","node_ids":20,"attach_to":"$node.10,$node.11"},{"labels":"learn","node_ids":31,"attach_to":"$node.10,$node.11,$node.20"},{"labels":"learn","node_ids":32,"attach_to":"$node.10,$node.11,$node.20"},{"labels":"learn","node_ids":33,"attach_to":"$node.10,$node.11,$node.20"}]}'
