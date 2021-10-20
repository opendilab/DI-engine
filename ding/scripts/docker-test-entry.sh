#!/usr/bin/env bash

CONTAINER_ID=$(docker run --rm -d opendilab/ding:nightly tail -f /dev/null)

trap "docker rm -f $CONTAINER_ID" EXIT

docker exec $CONTAINER_ID rm -rf /ding &&
  docker cp $(pwd) ${CONTAINER_ID}:/ding &&
  docker exec -it $CONTAINER_ID /ding/ding/scripts/docker-test.sh
