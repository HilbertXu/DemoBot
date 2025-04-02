set -e
cd Segment-and-Track-Anything
bash script/install.sh
bash script/download_ckpt.sh
pip install 'fsspec<2023.6.0'