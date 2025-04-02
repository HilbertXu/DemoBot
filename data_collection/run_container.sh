#!/bin/bash

DATA_DIR=../data
DOCKER_DATA_DIR=/workspace/data
ASSETS_DIR=../assets
DOCKER_ASSETS_DIR=/workspace/assets
DOCKER_NAME=$1


get_abs_path() {
    (cd "$1" && pwd)
}

xhost +local:root

docker run --gpus all --rm -it --network host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix/:/tmp/.X11-unix \
  -e "ACCEPT_EULA=Y" \
  -w /workspace/data_collection/generator \
  -v /usr/share/vulkan/icd.d/:/usr/share/vulkan/icd.d/ \
  -v $(get_abs_path $DATA_DIR):$DOCKER_DATA_DIR \
  -v $(get_abs_path $ASSETS_DIR):$DOCKER_ASSETS_DIR \
  $DOCKER_NAME bash