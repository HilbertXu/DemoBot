#!/bin/bash

docker run --gpus all --rm -it --network host \
  -w /workspace/data_collection/generator/Segment-and-Track-Anything \
  demobot-data-collection \
  /opt/conda/envs/sam-track/bin/python app.py