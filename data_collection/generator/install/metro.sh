set -e
cd MeshTransformer

# setup folders
mkdir -p ./models  # pre-trained models
mkdir -p ./datasets  # datasets
mkdir -p ./predictions  # prediction outputs


pip install -r requirements.txt
pip install git+https://github.com/mattloper/chumpy
python setup.py build develop
pip install torch torchvision torchaudio --force-reinstall  --extra-index-url https://download.pytorch.org/whl/cu116
pip install numpy==1.22.1 gdown
pip install ./manopth/.

# setup files
mkdir -p models/hrnet
mkdir -p models/metro_release

cd models/hrnet
gdown https://drive.google.com/uc\?id\=1liXnuJdRhEGoBQWebRPRFvUBDW6INiYT
cd ../metro_release
gdown https://drive.google.com/uc\?id\=1fNI7xo9SFwvyRiKZTIUpHXDdxV-mz1j_

