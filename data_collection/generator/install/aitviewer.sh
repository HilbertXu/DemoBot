#!/bin/bash
set -e

pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install easydict matplotlib chumpy loguru opencv-python-headless numpy

cd pytorch3d
python setup.py install
cd ..

cd smplx
python setup.py install
pip install aitviewer==1.13.0
pip install numpy==1.23.5
cd ..