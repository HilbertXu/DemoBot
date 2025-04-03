#!/bin/bash

DATA_DIR=../data
DOCKER_DATA_DIR=/workspace/data
ASSETS_DIR=../assets
DOCKER_ASSETS_DIR=/workspace/assets

SEQ_NAME=$1
OBJECT_MESH=$2

get_abs_path() {
    (cd "$1" && pwd)
}

xhost +local:root

docker run --gpus all --rm -it --network host \
  -w /workspace/data_collection/generator \
  -v $(get_abs_path $DATA_DIR):$DOCKER_DATA_DIR \
  -v $(get_abs_path $ASSETS_DIR):$DOCKER_ASSETS_DIR \
  demobot-data-collection \
  /opt/conda/envs/hold/bin/python scripts/downsample_files.py --data_dir $DOCKER_DATA_DIR --seq_name $SEQ_NAME --keep_interval 5



docker run --gpus all --rm -it --network host \
  -w /workspace/data_collection/generator \
  -v $(get_abs_path $DATA_DIR):$DOCKER_DATA_DIR \
  -v $(get_abs_path $ASSETS_DIR):$DOCKER_ASSETS_DIR \
  demobot-data-collection \
  /opt/conda/envs/hold/bin/python scripts/validate_masks.py --data_dir $DOCKER_DATA_DIR --seq_name $SEQ_NAME


docker run --gpus all --rm -it --network host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix/:/tmp/.X11-unix \
  -w /workspace/data_collection/generator/hand_detector.d2 \
  -v $(get_abs_path $DATA_DIR):$DOCKER_DATA_DIR \
  -v $(get_abs_path $ASSETS_DIR):$DOCKER_ASSETS_DIR \
  demobot-data-collection \
  /opt/conda/envs/100doh/bin/python crop_images.py --data_dir $DOCKER_DATA_DIR --seq_name $SEQ_NAME --scale 1.5 --min_size 256 --max_size 700



docker run --gpus all --rm -it --network host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix/:/tmp/.X11-unix \
  -w /workspace/data_collection/generator/MeshTransformer \
  -v $(get_abs_path $DATA_DIR):$DOCKER_DATA_DIR \
  -v $(get_abs_path $ASSETS_DIR):$DOCKER_ASSETS_DIR \
  demobot-data-collection \
  /opt/conda/envs/metro/bin/python metro/tools/end2end_inference_handmesh.py  --resume_checkpoint models/metro_release/metro_hand_state_dict.bin --image_file_or_path $DOCKER_DATA_DIR/$SEQ_NAME/processed/crop_image


docker run --gpus all --rm -it --network host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix/:/tmp/.X11-unix \
  -w /workspace/data_collection/generator \
  -v $(get_abs_path $DATA_DIR):$DOCKER_DATA_DIR \
  -v $(get_abs_path $ASSETS_DIR):$DOCKER_ASSETS_DIR \
  demobot-data-collection \
  /opt/conda/envs/hold/bin/python scripts/register_mano.py --data_dir $DOCKER_DATA_DIR --seq_name $SEQ_NAME --save_mesh --use_beta_loss



docker run --gpus all --rm -it --network host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix/:/tmp/.X11-unix \
  -w /workspace/data_collection/generator \
  -v $(get_abs_path $DATA_DIR):$DOCKER_DATA_DIR \
  -v $(get_abs_path $ASSETS_DIR):$DOCKER_ASSETS_DIR \
  demobot-data-collection \
  /opt/conda/envs/hold/bin/python scripts/validate_metro.py --data_dir $DOCKER_DATA_DIR --seq_name $SEQ_NAME


docker run --gpus all --rm -it --network host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix/:/tmp/.X11-unix \
  -w /workspace/data_collection/generator \
  -v $(get_abs_path $DATA_DIR):$DOCKER_DATA_DIR \
  -v $(get_abs_path $ASSETS_DIR):$DOCKER_ASSETS_DIR \
  demobot-data-collection \
  /opt/conda/envs/hold/bin/python scripts/project_object_model.py --data_dir $DOCKER_DATA_DIR --seq_name $SEQ_NAME --object_mesh $DOCKER_ASSETS_DIR/$OBJECT_MESH


docker run --gpus all --rm -it --network host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix/:/tmp/.X11-unix \
  -w /workspace/data_collection/generator \
  -v $(get_abs_path $DATA_DIR):$DOCKER_DATA_DIR \
  -v $(get_abs_path $ASSETS_DIR):$DOCKER_ASSETS_DIR \
  demobot-data-collection \
  /opt/conda/envs/hold/bin/python scripts/align_hands_object.py --data_dir $DOCKER_DATA_DIR --seq_name $SEQ_NAME --object_mesh $DOCKER_ASSETS_DIR/$OBJECT_MESH --mode h


docker run --gpus all --rm -it --network host \
  -e DISPLAY=$DISPLAY \
  -w /workspace/data_collection/generator \
  -v /tmp/.X11-unix/:/tmp/.X11-unix \
  -v /usr/share/vulkan/icd.d/:/usr/share/vulkan/icd.d/ \
  -v $(get_abs_path $DATA_DIR):$DOCKER_DATA_DIR \
  -v $(get_abs_path $ASSETS_DIR):$DOCKER_ASSETS_DIR \
  demobot-data-collection \
  /opt/conda/envs/metro/bin/python scripts/retargeting.py --mano_root $DOCKER_ASSETS_DIR/mano --scene_dir $DOCKER_DATA_DIR/$SEQ_NAME --config retargeting_config/allegro_retarget_base.yml --assets_root $DOCKER_ASSETS_DIR --hand_base base_link


## For iphone data
## extrinsic 