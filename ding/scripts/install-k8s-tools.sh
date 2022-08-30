#!/usr/bin/env bash

set -e

ROOT_DIR="$(dirname "$0")"
: ${USE_SUDO:="true"}

# runs the given command as root (detects if we are root already)
runAsRoot() {
    local CMD="$*"

    if [ $EUID -ne 0 -a $USE_SUDO = "true" ]; then
        CMD="sudo $CMD"
    fi

    $CMD
}

# install k3d
curl -s https://raw.githubusercontent.com/rancher/k3d/main/install.sh | TAG=v4.4.8 bash

# install kubectl
if [[ $(which kubectl) == "" ]]; then
    echo "Installing kubectl..."
    curl -LO https://dl.k8s.io/release/v1.21.3/bin/linux/amd64/kubectl
    chmod +x kubectl
    runAsRoot mv kubectl /usr/local/bin/kubectl
fi
