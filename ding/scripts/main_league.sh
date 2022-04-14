#!/usr/bin/env bash

BASEDIR=$(dirname "$0")/../entry

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

# ditask --package $BASEDIR \
#   --main main_league.main \
#   --parallel-workers 3 \
#   --protocol tcp \
#   --address 127.0.0.1 \
#   --ports 50525 \
#   --node-ids 10 \
#   --topology alone \
#   --labels learn \
#   --attach-to tcp://127.0.0.1:50515 &

# ditask --package $BASEDIR \
#   --main main_league.main \
#   --parallel-workers 1 \
#   --address 127.0.0.1 \
#   --protocol tcp \
#   --ports 50535 \
#   --node-ids 20 \
#   --topology alone \
#   --labels evaluate \
#   --attach-to tcp://127.0.0.1:50515,tcp://127.0.0.1:50525,tcp://127.0.0.1:50526,tcp://127.0.0.1:50527 &

sleep 10000
