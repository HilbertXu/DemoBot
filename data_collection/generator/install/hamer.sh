set -e
cd hamer

# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
mamba install -c conda-forge pybind11
mamba install -c conda-forge gxx
mamba install -c anaconda gcc_linux-64
mamba upgrade -c conda-forge --all

pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'
pip install -e ".[all]" 
pip install -v -e third-party/ViTPose
pip install pillow==9.1.0
# bash fetch_demo_data.sh
cd ..
mkdir -p hamer/_DATA/data/mano
cp -r ../code/body_models/* hamer/_DATA/data/mano