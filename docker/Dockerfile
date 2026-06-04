ARG CUDA_VERSION=12.4.1
ARG OS_VERSION=22.04

### build image ###
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu${OS_VERSION} AS build

RUN echo 'Acquire::Retries "3";' > /etc/apt/apt.conf.d/80retries

ENV DEBIAN_FRONTEND=noninteractive

# install python3-pip (and make `python` point to python3)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        python-is-python3 \
        curl \
        gnupg2 \
        lsb-release \
        ca-certificates \
        git \
        sudo \
        locales \
        software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# install system dependencies that lerobot needs to pip-install cleanly
# (borrowed from huggingface/lerobot's Dockerfile.user)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        pkg-config \
        ninja-build \
        ffmpeg \
        libglib2.0-0 \
        libegl1-mesa-dev \
        libgl1 \
        libusb-1.0-0-dev \
        speech-dispatcher \
        libgeos-dev \
        portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# install ros humble desktop
RUN locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && add-apt-repository universe \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
         -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
         > /etc/apt/sources.list.d/ros2.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-desktop \
        ros-dev-tools \
        python3-colcon-common-extensions \
        python3-rosdep \
    && rosdep init \
    && rm -rf /var/lib/apt/lists/*

# realsense python bindings for depth camera support
RUN pip install --no-cache-dir pyrealsense2


### dev ###
FROM build AS dev

# create "ubuntu" user (UID 1000) with passwordless sudo
RUN useradd --create-home --shell /bin/bash --uid 1000 ubuntu \
    && usermod -aG sudo ubuntu \
    && echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu

USER ubuntu
WORKDIR /home/ubuntu

# run rosdep update as the normal user so caches land in the right place
RUN rosdep update || true

# general dev tools
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    vim \
    nano \
    less \
    tmux

# environment settings in the bashrc
RUN echo "export PATH=\"$HOME/.local/bin:$PATH\"" >> /home/ubuntu/.bashrc && \
    echo "source /opt/ros/humble/setup.bash" >> /home/ubuntu/.bashrc  && \
    echo "source /home/ubuntu/techin517/ros2_ws/install/setup.bash" >> /home/ubuntu/.bashrc && \
    echo "export ROS_LOCALHOST_ONLY=1" >> /home/ubuntu/.bashrc && \
    echo "export HF_USER=ubuntu" >> /home/ubuntu/.bashrc && \
    echo "export XDG_RUNTIME_DIR=/tmp/runtime-ubuntu" >> /home/ubuntu/.bashrc

# Default shell
CMD ["/bin/bash"]