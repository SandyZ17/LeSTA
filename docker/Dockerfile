FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    ROS_DISTRO=humble

# Use USTC mirrors for Ubuntu, ROS, and regular Python packages.
RUN printf '%s\n' \
    'deb https://mirrors.ustc.edu.cn/ubuntu/ jammy main restricted universe multiverse' \
    'deb https://mirrors.ustc.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse' \
    'deb https://mirrors.ustc.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse' \
    'deb https://mirrors.ustc.edu.cn/ubuntu/ jammy-security main restricted universe multiverse' \
    > /etc/apt/sources.list \
    && printf '%s\n' \
    '[global]' \
    'index-url = https://pypi.mirrors.ustc.edu.cn/simple' \
    'trusted-host = pypi.mirrors.ustc.edu.cn' \
    > /etc/pip.conf

# ROS 2 Humble targets Ubuntu 22.04 (Jammy).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg2 lsb-release git wget unzip \
    build-essential cmake ninja-build pkg-config \
    software-properties-common \
    python3 python3-dev python3-pip python3-venv \
    && curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
       -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] https://mirrors.ustc.edu.cn/ros2/ubuntu jammy main" \
       > /etc/apt/sources.list.d/ros2.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       ros-humble-ros-base \
       python3-colcon-common-extensions \
       ros-humble-grid-map-core \
       ros-humble-grid-map-ros \
       ros-humble-grid-map-msgs \
       ros-humble-pcl-ros \
       ros-humble-pcl-conversions \
       ros-humble-tf2-ros \
       ros-humble-tf2-eigen \
       ros-humble-tf2-geometry-msgs \
       ros-humble-rviz2 \
       ros-humble-rosidl-default-generators \
       ros-humble-rosbag2-py \
       ros-humble-rosbag2-storage-default-plugins \
    && rm -rf /var/lib/apt/lists/* \
    && python3 --version

WORKDIR /workspace/lesta_ws
COPY . /workspace/lesta_ws/src/LeSTA

# The PyTorch image already contains Python PyTorch and its matching LibTorch
# headers/libraries. Expose that installation at a stable path for CMake.
RUN TORCH_ROOT="$(python3 -c 'import torch; print(torch.__path__[0])')" \
 && ln -s "${TORCH_ROOT}" /opt/torch \
 && export CMAKE_PREFIX_PATH="/opt/torch:${CMAKE_PREFIX_PATH}" \
 && export LD_LIBRARY_PATH="/opt/torch/lib:${LD_LIBRARY_PATH}" \
 && python3 -m pip install --no-cache-dir catkin-pkg empy==3.3.4 lark \
 && python3 -m pip install --no-cache-dir -e /workspace/lesta_ws/src/LeSTA/pylesta \
 && source /opt/ros/${ROS_DISTRO}/setup.bash \
 && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

RUN echo 'source /opt/ros/${ROS_DISTRO}/setup.bash' >> /root/.bashrc \
 && echo 'source /workspace/lesta_ws/install/setup.bash' >> /root/.bashrc

CMD ["/bin/bash"]
