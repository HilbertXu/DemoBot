set -e
echo 'Setup metro env :: dex-retargeting'

cd dex-retargeting
pip install -e .
pip install typo tqdm opencv-python mediapipe sapien==3.0.0b0 loguru