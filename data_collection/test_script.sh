#!/bin/bash

# Show help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo "Required:"
    echo "  --seq_name NAME"
    echo "  --right_object_mesh FILE"
    echo "  --left_object_mesh FILE"
    echo "  --right_object_keypoints FILE"
    echo "  --left_object_keypoints FILE"
    echo "  --hand_estimator NAME       (e.g., metro or wilor)"
    echo "  --hand_type NAME"
    echo "Optional:"
    echo "  --refine_object_pose 1|0"
    echo "  --refine_object_pose_task TASK"
    echo "  --help"
    exit 1
}

# If no arguments or --help
if [ "$#" -eq 0 ]; then show_help; fi

# Use getopt to parse
OPTS=$(getopt -o "" -l seq_name:,right_object_mesh:,left_object_mesh:,right_object_keypoints:,left_object_keypoints:,hand_estimator:,hand_type:,refine_object_pose:,refine_object_pose_task:,help -- "$@")
eval set -- "$OPTS"

# Default values
REFINE_OBJECT_POSE=0
REFINE_OBJECT_POSE_TASK=""

# Parse
while true; do
  case "$1" in
    --seq_name) SEQ_NAME="$2"; shift 2 ;;
    --right_object_mesh) RIGHT_OBJECT_MESH="$2"; shift 2 ;;
    --left_object_mesh) LEFT_OBJECT_MESH="$2"; shift 2 ;;
    --right_object_keypoints) RIGHT_OBJECT_KEYPOINTS="$2"; shift 2 ;;
    --left_object_keypoints) LEFT_OBJECT_KEYPOINTS="$2"; shift 2 ;;
    --hand_estimator) HAND_ESTIMATOR="$2"; shift 2 ;;
    --hand_type) HAND_TYPE="$2"; shift 2 ;;
    --refine_object_pose) REFINE_OBJECT_POSE="$2"; shift 2 ;;
    --refine_object_pose_task) REFINE_OBJECT_POSE_TASK="$2"; shift 2 ;;
    --help) show_help ;;
    --) shift; break ;;
    *) echo "Unknown option $1"; show_help ;;
  esac
done