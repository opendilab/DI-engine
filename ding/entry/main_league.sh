#!/usr/bin/env bash

BASEDIR=$(dirname "$0")

kill_descendant_processes() {
  local pid="$1"
  local and_self="${2:-false}"
  if children="$(pgrep -P "$pid")"; then
    for child in $children; do
      kill_descendant_processes "$child" true
    done
  fi
  if [[ "$and_self" == true ]]; then
    kill "$pid"
  fi
}

trap "kill_descendant_processes $$" EXIT

ditask --package $BASEDIR \
  --main main_league.main \
  --parallel-workers 1 \
  --protocol tcp \
  --address 127.0.0.1 \
  --ports 50515 \
  --node-ids 0 \
  --topology alone \
  --labels league,collect &

ditask --package $BASEDIR \
  --main main_league.main \
  --parallel-workers 3 \
  --protocol tcp \
  --address 127.0.0.1 \
  --ports 50525 \
  --node-ids 10 \
  --topology alone \
  --labels learn \
  --attach-to tcp://127.0.0.1:50515 &

ditask --package $BASEDIR \
  --main main_league.main \
  --parallel-workers 1 \
  --address 127.0.0.1 \
  --protocol tcp \
  --ports 50535 \
  --node-ids 20 \
  --topology alone \
  --labels evaluate \
  --attach-to tcp://127.0.0.1:50515,tcp://127.0.0.1:50525,tcp://127.0.0.1:50526,tcp://127.0.0.1:50527 &

############# Start slurm test #############

# srun -p Cerebra_Share --quotatype=reserved  --mpi=pmi2 --gres=gpu:4 -n4 --ntasks-per-node=2 python3 -u test.py
export HOSTNAME='SH-IDC1-10-5-38-31'              # 宿主机，无意义
export SLURM_NTASKS='4'                           # 参数 n，总进程/任务数
export SLURM_NTASKS_PER_NODE='2'                  # 参数 ntasks-per-node，每个节点的进程数
export SLURM_NODELIST='SH-IDC1-10-5-38-[190,215]' # 所有节点
export SLURM_SRUN_COMM_PORT='42932'               # 哪些可用端口？
export SLURM_TOPOLOGY_ADDR='SH-IDC1-10-5-38-215'  # 当前节点名
export SLURM_NODEID='1'                           # 节点顺序，从 0 开始
export SLURM_PROCID='2'                           # 进程顺序，从 0 开始，实际启动顺序可能与数字顺序不同
export SLURM_LOCALID='0'                          # 本地顺序，从 0 开始，最大 ntasks-per-node - 1
export SLURM_GTIDS='2,3'                          # 在当前进程上启动的 procid
export SLURMD_NODENAME='SH-IDC1-10-5-38-215'      # 当前节点名
ditask --package $BASEDIR \
  --main main_league.main \
  --platform-spec '{"type":"slurm","labels":""}' &
############# Finish slurm test #############

sleep 10000
