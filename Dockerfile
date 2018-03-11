FROM nvidia/opengl:1.0-glvnd-devel

##################
#  Install CUDA  #
##################

RUN NVIDIA_GPGKEY_SUM=d1be581509378368edeec8c1eb2958702feedf3bc3d17011adbf24efacce4ab5 && \
    NVIDIA_GPGKEY_FPR=ae09fe4bbd223a84b2ccfce3f60f4b3d7fa2af80 && \
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub && \
    apt-key adv --export --no-emit-version -a $NVIDIA_GPGKEY_FPR | tail -n +5 > cudasign.pub && \
    echo "$NVIDIA_GPGKEY_SUM  cudasign.pub" | sha256sum -c --strict - && rm cudasign.pub && \
    echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

ENV CUDA_VERSION 9.0.176

ENV CUDA_PKG_VERSION 9-0=$CUDA_VERSION-1
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-cudart-$CUDA_PKG_VERSION && \
    ln -s cuda-9.0 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

# nvidia-docker 1.0
LABEL com.nvidia.volumes.needed="nvidia_driver"
LABEL com.nvidia.cuda.version="${CUDA_VERSION}"

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=9.0"

ENV NCCL_VERSION 2.1.15

RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-libraries-$CUDA_PKG_VERSION \
        libnccl2=$NCCL_VERSION-1+cuda9.0 && \
    rm -rf /var/lib/apt/lists/*


RUN apt-get update

RUN apt-get install -y \
    libglu1-mesa \
    vtk6 \
    libvtk6-dev \
    mayavi2 \
    libboost-all-dev \
    python-dev \
    python3-dev \
    git \ 
    cmake \ 
    g++ \ 
    gdb \ 
    python-dbg \
    python3-dbg \
    build-essential

RUN apt-get install -y wget

#################################
#  Install Boost on Python 3.5  #
#################################

RUN apt-get install libboost-all-dev
WORKDIR /opt/boost
RUN wget https://dl.bintray.com/boostorg/release/1.65.1/source/boost_1_65_1.tar.gz
RUN tar -xvzf boost_1_65_1.tar.gz
WORKDIR /opt/boost/boost_1_65_1
RUN ./bootstrap.sh --with-python-version=3.5 --with-python=/usr/bin/python3.5 --with-python-root=/usr/local/lib/python3.5
RUN ./b2
RUN ./b2 install

#####################
#  Install GCC 4.9  #
#####################

RUN apt-get install -y software-properties-common \
    python-software-properties
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test
RUN apt-get update
RUN apt-get -y install gcc-6 gcc-4.9
RUN apt-get -y install g++-6 g++-4.9
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 60 --slave /usr/bin/g++ g++ /usr/bin/g++-6
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 50 --slave /usr/bin/g++ g++ /usr/bin/g++-4.9
RUN update-alternatives --set gcc /usr/bin/gcc-4.9

#################
#  Install VTK  #
#################

RUN apt-get install -y \
    vtk6 \
    python-vtk

RUN apt-get install -y --no-install-recommends \
        mesa-utils

RUN git clone --depth=1 --branch=v8.1.0 https://gitlab.kitware.com/vtk/vtk.git && \
    mkdir vtk/build && cd vtk/build && \
    cmake .. -DVTK_RENDERING_BACKEND=OpenGL2 -DVTK_USE_X=OFF -DBUILD_TESTING=OFF -DVTK_WRAP_PYTHON=ON && \
    make -j"$(nproc)" install && \
    cd / && rm -rf /vtk


###################
#  Upgrade CMAKE  #
###################

WORKDIR /opt/cmake
RUN wget https://cmake.org/files/v3.10/cmake-3.10.2.tar.gz
RUN tar -xvzf cmake-3.10.2.tar.gz
WORKDIR /opt/cmake/cmake-3.10.2
RUN cmake .
RUN make -j"$(nproc)"
RUN make -j"$(nproc)" install

#  Install CUDA Samples

RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-samples-$CUDA_PKG_VERSION && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /usr/local/cuda/samples

RUN make -j"$(nproc)" -k || true

RUN export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
RUN export LD_LIBRARY_PATH=/usr/local/cuda/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

#########################
#  Python Requirements  #
#########################
RUN apt update -y
RUN apt install -y \
        python-pip \
        python3-pip
RUN python -m pip install --upgrade pip setuptools wheel
RUN python3 -m pip install --upgrade pip setuptools wheel


####################
#  Install Netgen  #
####################

RUN apt-add-repository -y universe 

RUN add-apt-repository -y ppa:ngsolve/ngsolve
RUN apt-get update -y

RUN apt-get install -y ngsolve

############
#  OCTMPS  #
############

COPY . /opt/octmps

WORKDIR /opt/octmps
RUN mkdir build
WORKDIR /opt/octmps/build
RUN cmake ..
RUN make -j "$(nproc)"
RUN make -j "$(nproc)" install
ENV PYTHONPATH $PYTHONPATH:/opt/octmps
WORKDIR /opt/octmps

RUN python -m pip install -r requirements.txt
RUN python3 -m pip install -U numpy
#RUN python3 -m pip install -r requirements.txt

#########
#  X11  #
#########
RUN apt-get update
RUN apt-get install -qqy x11-apps
ENV DISPLAY :0