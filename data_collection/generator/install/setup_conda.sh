#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh

cd generator
echo 'Setup sam-track env'
conda activate sam-track
bash ./install/sam.sh
conda deactivate

echo 'Setup 100doh env'
conda activate 100doh
bash ./install/100doh.sh 
bash ./install/foundationpose.sh
conda deactivate

echo 'Setup metro env'
conda activate metro
bash ./install/metro.sh
conda deactivate

echo 'Setup WiLoR env'
conda activate wilor
bash ./install/wilor.sh
bash ./install/retargeting.sh
conda deactivate

echo 'Setup hold env'
conda activate hold

python -m pip install pip==24.0 --upgrade
pip install -r requirements.txt
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install ninja
mkdir submodules 
cd submodules

echo 'Setup hold env :: pytorch3d' 
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
git checkout 35badc08
python setup.py install

echo 'Setup hold env :: kaolin' 
git clone --recursive https://github.com/NVIDIAGameWorks/kaolin
cd kaolin
git checkout v0.10.0
python setup.py install

echo 'Setup hold env :: smplx' 
git clone https://github.com/zc-alexfan/smplx.git
cd smplx
git checkout 6675c3da8
python setup.py install
cd ..
pip install setuptools==59.5.0
pip install numpy==1.23.5
pip install scikit-image==0.18.1
pip install 'fsspec<2023.6.0'
conda deactivate

echo 'Setup aitviewer env'
conda activate aitviewer
bash ./install/retargeting.sh
conda deactivate 
cd ..
rm -rf submodules



