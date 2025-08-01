FROM nvidia/cudagl:11.1.1-devel-ubuntu20.04

ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute,display
ENV TZ=US/Pacific
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update --fix-missing && \
    apt-get install -y libgtk2.0-dev && \
    apt-get install -y ffmpeg && \
    apt-get install -y wget bzip2 ca-certificates curl unzip git vim tmux g++ gcc build-essential cmake checkinstall gfortran libjpeg8-dev libtiff5-dev pkg-config yasm libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev libxine2-dev libv4l-dev qt5-default libgtk2.0-dev libtbb-dev libatlas-base-dev libfaac-dev libmp3lame-dev libtheora-dev libvorbis-dev libxvidcore-dev libopencore-amrnb-dev libopencore-amrwb-dev x264 v4l-utils libprotobuf-dev protobuf-compiler libgoogle-glog-dev libgflags-dev libgphoto2-dev libhdf5-dev doxygen libflann-dev libboost-all-dev proj-data libproj-dev libyaml-cpp-dev cmake-curses-gui libzmq3-dev freeglut3-dev


RUN cd / && git clone https://github.com/pybind/pybind11 &&\
    cd pybind11 && git checkout v2.10.0 &&\
    mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DPYBIND11_INSTALL=ON -DPYBIND11_TEST=OFF &&\
    make -j6 && make install -j8 &&\
    cd / && rm -rf pybind


RUN cd / && wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz &&\
    tar xvzf ./eigen-3.4.0.tar.gz &&\
    cd eigen-3.4.0 &&\
    mkdir build &&\
    cd build &&\
    cmake .. &&\
    make install -j8 &&\
    cd / && rm -rf eigen-3.4.0 

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "--login", "-c"]

RUN cd / && wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    /bin/bash /miniconda.sh -b -p /opt/conda &&\
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh &&\
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc &&\
    /bin/bash -c "source ~/.bashrc" &&\
    rm /miniconda.sh

RUN /opt/conda/bin/conda tos accept
   
RUN /opt/conda/bin/conda create --name 100doh python=3.8 &&\
    /opt/conda/bin/conda create --name metro python=3.8 &&\
    /opt/conda/bin/conda create --name sam-track python=3.8 &&\
    /opt/conda/bin/conda create --name hold python=3.8 &&\
    /opt/conda/bin/conda create --name aitviewer python=3.8 &&\
    /opt/conda/bin/conda create --name wilor python=3.10 &&\
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc &&\
    echo "alias pyhold=/opt/conda/envs/hold/bin/python" >> ~/.bashrc &&\
    echo "alias pymetro=/opt/conda/envs/metro/bin/python" >> ~/.bashrc &&\
    echo "alias pysam=/opt/conda/envs/sam-track/bin/python" >> ~/.bashrc &&\
    echo "alias pydoh=/opt/conda/envs/100doh/bin/python" >> ~/.bashrc &&\
    echo "alias pyait=/opt/conda/envs/aitviewer/bin/python" >> ~/.bashrc &&\
    echo "alias pywilor=/opt/conda/envs/wilor/bin/python" >> ~/.bashrc &&\
    echo "export TZ=$(date +%Z)" >> ${HOME}/.bashrc


ENV PATH $PATH:/opt/conda/envs/my/bin

RUN conda init --all

RUN touch /bin/nvidia-smi && \
    touch /bin/nvidia-debugdump && \
    touch /bin/nvidia-persistenced && \
    touch /bin/nvidia-cuda-mps-control && \
    touch /bin/nvidia-cuda-mps-server && \
    touch /etc/localtime && \
    mkdir -p /var/run/nvidia-persistenced && \
    touch /var/run/nvidia-persistenced/socket


WORKDIR /workspace

COPY data_collection /workspace/data_collection

RUN cd data_collection/generator &&\
    bash ./install/setup_conda.sh

RUN conda clean --all -f -y
