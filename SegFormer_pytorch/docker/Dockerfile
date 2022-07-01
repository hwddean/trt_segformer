#ARG PYTORCH="1.7.0"
#ARG CUDA="10.2"
#ARG CUDNN="7.5"


#FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

COPY opencv_python-4.5.5.64-cp36-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl  /tmp
RUN pip install /tmp/opencv_python-4.5.5.64-cp36-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Install mmsegmentation
RUN conda clean --all
#RUN git config --global http.proxy http://192.168.50.84:7890
RUN pip install mmcv-full==latest+torch1.6.0+cu101 -f https://download.openmmlab.com/mmcv/dist/index.html
#RUN git clone https://github.com/open-mmlab/mmsegmenation.git /mmsegmentation
COPY mmsegmentation/. /mmsegmentation
WORKDIR /mmsegmentation
RUN pip install -r requirements.txt
RUN apt update
RUN apt install libgl1-mesa-glx
WORKDIR /app
pip install -e . 
#RUN pip install --no-cache-dir -e .
