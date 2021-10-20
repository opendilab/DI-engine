#!/usr/bin/env bash

if [ ! -f /.dockerenv ]; then
  echo "This script should be executed in docker container"
  exit 1
fi

pip install --ignore-installed 'PyYAML<6.0'
pip install -e .[test,k8s] &&
  ./ding/scripts/install-k8s-tools.sh &&
  make test
