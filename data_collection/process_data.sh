#!/bin/bash

DATA_DIR=../data
DOCKER_DATA_DIR=/workspace/data
ASSETS_DIR=../assets
DOCKER_ASSETS_DIR=/workspace/assets

SEQ_NAME=$1
RIGHT_OBJECT_MESH=$2
LEFT_OBJECT_MESH=$3
RIGHT_OBJECT_KEYPOINTS=$4
LEFT_OBJECT_KEYPOINTS=$5
HAND_ESTIMATOR=$6
HAND_TYPE=$7
REFINE_OBJECT_POSE=$8
REFINE_OBJECT_POSE_TASK=$9

start=$(date +%s)
get_abs_path() {
    (cd "$1" && pwd)
}

xhost +local:root


docker run --gpus all --rm -it --network host \
  -w /workspace/data_collection/generator \
  -v $(get_abs_path $DATA_DIR):$DOCKER_DATA_DIR \
  -v $(get_abs_path $ASSETS_DIR):$DOCKER_ASSETS_DIR \
  demobot-data-collection \
  /opt/conda/envs/hold/bin/python scripts/validate_masks.py --data_dir $DOCKER_DATA_DIR --seq_name $SEQ_NAME


echo "Use hand estimator: $HAND_ESTIMATOR"
if [ "$HAND_ESTIMATOR" = "metro" ]; then
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


elif [ "$HAND_ESTIMATOR" = "wilor" ]; then
  docker run --gpus all --rm -it --network host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -w /workspace/data_collection/generator/WiLoR \
    -v $(get_abs_path $DATA_DIR):$DOCKER_DATA_DIR \
    -v $(get_abs_path $ASSETS_DIR):$DOCKER_ASSETS_DIR \
    demobot-data-collection \
    /opt/conda/envs/wilor/bin/python demo.py  --data_dir $DOCKER_DATA_DIR --seq_name $SEQ_NAME --save_mesh

  docker run --gpus all --rm -it --network host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -w /workspace/data_collection/generator \
    -v $(get_abs_path $DATA_DIR):$DOCKER_DATA_DIR \
    -v $(get_abs_path $ASSETS_DIR):$DOCKER_ASSETS_DIR \
    demobot-data-collection \
    /opt/conda/envs/hold/bin/python scripts/register_mano_wilor.py --data_dir $DOCKER_DATA_DIR --seq_name $SEQ_NAME --save_mesh --use_beta_loss --hand_type $HAND_TYPE

  docker run --gpus all --rm -it --network host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix/:/tmp/.X11-unix \
    -w /workspace/data_collection/generator \
    -v $(get_abs_path $DATA_DIR):$DOCKER_DATA_DIR \
    -v $(get_abs_path $ASSETS_DIR):$DOCKER_ASSETS_DIR \
    demobot-data-collection \
    /opt/conda/envs/hold/bin/python scripts/validate_wilor.py --data_dir $DOCKER_DATA_DIR --seq_name $SEQ_NAME --hand_type $HAND_TYPE

else 
  echo "Unsupported hand estimator: $HAND_ESTIMATOR"
fi


docker run --gpus all --rm -it --network host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix/:/tmp/.X11-unix \
  -w /workspace/data_collection/generator \
  -v $(get_abs_path $DATA_DIR):$DOCKER_DATA_DIR \
  -v $(get_abs_path $ASSETS_DIR):$DOCKER_ASSETS_DIR \
  demobot-data-collection \
  /opt/conda/envs/hold/bin/python scripts/project_object_model.py --data_dir $DOCKER_DATA_DIR --seq_name $SEQ_NAME --right_object_mesh $DOCKER_ASSETS_DIR/$RIGHT_OBJECT_MESH --left_object_mesh $DOCKER_ASSETS_DIR/$LEFT_OBJECT_MESH


docker run --gpus all --rm -it --network host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix/:/tmp/.X11-unix \
  -w /workspace/data_collection/generator \
  -v $(get_abs_path $DATA_DIR):$DOCKER_DATA_DIR \
  -v $(get_abs_path $ASSETS_DIR):$DOCKER_ASSETS_DIR \
  demobot-data-collection \
  /opt/conda/envs/hold/bin/python scripts/align_hands_object.py --data_dir $DOCKER_DATA_DIR --seq_name $SEQ_NAME --right_object_mesh $DOCKER_ASSETS_DIR/$RIGHT_OBJECT_MESH --left_object_mesh $DOCKER_ASSETS_DIR/$LEFT_OBJECT_MESH --mode h

  if [ "$REFINE_OBJECT_POSE" = "1" ]; then
    docker run --gpus all --rm -it --network host \
      -e DISPLAY=$DISPLAY \
      -e QT_X11_NO_MITSHM=1 \
      -v /tmp/.X11-unix/:/tmp/.X11-unix \
      -w /workspace/data_collection/generator \
      -v $(get_abs_path $DATA_DIR):$DOCKER_DATA_DIR \
      -v $(get_abs_path $ASSETS_DIR):$DOCKER_ASSETS_DIR \
      --device=/dev/dri:/dev/dri \
      demobot-data-collection \
      /opt/conda/envs/hold/bin/python scripts/refine_object_poses.py --asset_dir $DOCKER_ASSETS_DIR --data_dir $DOCKER_DATA_DIR --seq_name $SEQ_NAME --right_object_keypoints $RIGHT_OBJECT_KEYPOINTS --left_object_keypoints $LEFT_OBJECT_KEYPOINTS --task insert --mode ro --num_frames_to_refine 3
  else 
    echo "Skip object pose refinement"
  fi

docker run --gpus all --rm -it --network host \
  -e DISPLAY=$DISPLAY \
  -w /workspace/data_collection/generator \
  -v /tmp/.X11-unix/:/tmp/.X11-unix \
  -v /usr/share/vulkan/icd.d/:/usr/share/vulkan/icd.d/ \
  -v $(get_abs_path $DATA_DIR):$DOCKER_DATA_DIR \
  -v $(get_abs_path $ASSETS_DIR):$DOCKER_ASSETS_DIR \
  demobot-data-collection \
  /opt/conda/envs/metro/bin/python scripts/retargeting.py --mano_root $DOCKER_ASSETS_DIR/mano --scene_dir $DOCKER_DATA_DIR/$SEQ_NAME --right_config retargeting_config/allegro_retarget_base_right.yml --left_config retargeting_config/allegro_retarget_base_left.yml --assets_root $DOCKER_ASSETS_DIR --hand_base base_link

# docker run --gpus all --rm -it --network host \
#   -e DISPLAY=$DISPLAY \
#   -w /workspace/data_collection/generator \
#   -v /tmp/.X11-unix/:/tmp/.X11-unix \
#   -v /usr/share/vulkan/icd.d/:/usr/share/vulkan/icd.d/ \
#   -v $(get_abs_path $DATA_DIR):$DOCKER_DATA_DIR \
#   -v $(get_abs_path $ASSETS_DIR):$DOCKER_ASSETS_DIR \
#   demobot-data-collection \
#   /opt/conda/envs/metro/bin/python scripts/retargeting.py --mano_root $DOCKER_ASSETS_DIR/mano --scene_dir $DOCKER_DATA_DIR/$SEQ_NAME --config retargeting_config/leap_hand_right.yml --assets_root $DOCKER_ASSETS_DIR --hand_base base


# docker run --gpus all --rm -it --network host \
#   -e DISPLAY=$DISPLAY \
#   -w /workspace/data_collection/generator \
#   -v /tmp/.X11-unix/:/tmp/.X11-unix \
#   -v /usr/share/vulkan/icd.d/:/usr/share/vulkan/icd.d/ \
#   -v $(get_abs_path $DATA_DIR):$DOCKER_DATA_DIR \
#   -v $(get_abs_path $ASSETS_DIR):$DOCKER_ASSETS_DIR \
#   demobot-data-collection \
#   /opt/conda/envs/metro/bin/python scripts/retargeting.py --mano_root $DOCKER_ASSETS_DIR/mano --scene_dir $DOCKER_DATA_DIR/$SEQ_NAME --config retargeting_config/panda_gripper.yml --assets_root $DOCKER_ASSETS_DIR --hand_base panda_hand

end=$(date +%s)
runtime=$((end - start))
echo "Script took $runtime seconds"

## For iphone data
## extrinsic 