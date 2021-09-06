#!/bin/bash

set -e

ROOT_DIR="$(dirname "$0")"

curl -s https://raw.githubusercontent.com/rancher/k3d/main/install.sh | TAG=v4.4.8 bash
