FROM ros:noetic-ros-base-focal

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN printf '%s\n' \
    'deb https://mirrors.ustc.edu.cn/ubuntu/ focal main restricted universe multiverse' \
    'deb https://mirrors.ustc.edu.cn/ubuntu/ focal-updates main restricted universe multiverse' \
    'deb https://mirrors.ustc.edu.cn/ubuntu/ focal-security main restricted universe multiverse' \
    > /etc/apt/sources.list \
    && printf '%s\n' \
    'deb [trusted=yes] https://mirrors.ustc.edu.cn/ros/ubuntu focal main' \
    > /etc/apt/sources.list.d/ros1.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       ros-noetic-velodyne-pointcloud \
       ros-noetic-nodelet \
       python3-rospy \
       python3-numpy \
       && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/data_preprocessing
COPY ros1 /workspace/data_preprocessing/ros1

ENTRYPOINT ["/bin/bash"]
