#!/usr/bin/env bash

if [ ! -f /.dockerenv ]; then
  echo "This script should be executed in docker container"
  exit 1
fi

cp -rf /ding /tmp/ding
cd /tmp/ding

pip install -e .[test,k8s] &&
  ./ding/scripts/install-k8s-tools.sh &&
  make test
