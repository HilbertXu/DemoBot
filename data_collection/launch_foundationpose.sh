#!/bin/bash

DATA_DIR=../data
DOCKER_DATA_DIR=/workspace/data
ASSETS_DIR=../assets
DOCKER_ASSETS_DIR=/workspace/assets
SEQ_NAME=$1
OBJECT_MESH_FILE=$2
MASK_FOLDER=$3
IGNORE_X_AXIS=${4:-0}
IGNORE_Y_AXIS=${5:-0}
IGNORE_Z_AXIS=${6:-0}

xhost +local:root

docker run --gpus all --rm -it --network host \
  -w /workspace/data_collection/generator/FoundationPose \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix/:/tmp/.X11-unix \
  -v "$(pwd)/$DATA_DIR":"$DOCKER_DATA_DIR" \
  -v "$(pwd)/$ASSETS_DIR":"$DOCKER_ASSETS_DIR" \
  -v "$(pwd)/../data_collection":/workspace/data_collection \
  foundationpose \
  /opt/conda/envs/my/bin/python run_demo.py --mesh_file $DOCKER_ASSETS_DIR/$OBJECT_MESH_FILE --test_scene_dir $DOCKER_DATA_DIR/$SEQ_NAME --mask_folder $MASK_FOLDER --est_refine_iter 8 --track_refine_iter 5 --debug 3 --ignore_x_axis $IGNORE_X_AXIS --ignore_y_axis $IGNORE_Y_AXIS --ignore_z_axis $IGNORE_Z_AXIS