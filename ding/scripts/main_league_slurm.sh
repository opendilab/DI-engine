#!/usr/bin/env bash

BASEDIR=$(dirname "$0")

# srun -p Cerebra_Share --quotatype=reserved --mpi=pmi2 -n6 --ntasks-per-node=3
ditask --package $BASEDIR/../entry --main main_league.main --platform-spec '
{
  "type": "slurm",
  "tasks": [
    {
      "labels": "league,collect",
      "node_ids": 10,
    },
    {
      "labels": "league,collect",
      "node_ids": 11
    },
    {
      "labels": "evaluate",
      "node_ids": 20,
      "attach_to": "$node.10,$node.11"
    },
    {
      "labels": "learn",
      "node_ids": 31,
      "attach_to": "$node.10,$node.11,$node.20"
    },
    {
      "labels": "learn",
      "node_ids": 32,
      "attach_to": "$node.10,$node.11,$node.20"
    },
    {
      "labels": "learn",
      "node_ids": 33,
      "attach_to": "$node.10,$node.11,$node.20"
    }
  ]
}'
