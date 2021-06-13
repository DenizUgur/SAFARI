FROM ros:melodic
LABEL description="SAFARI Image"

# These values will be overrided by `docker run --env <key>=<value>` command
ENV ROS_IP 127.0.0.1
ENV ROS_MASTER_URI http://127.0.0.1:11311

# Initialization
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends aptitude openssh-server curl ssh apt-transport-https software-properties-common screen vim && \
    rm -rf /var/lib/apt/lists/*

# Set root password
RUN echo 'root:root' | chpasswd

# Permit SSH root login
RUN sed -i 's/#*PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config

# Add ROS related packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-melodic-grid-map ros-melodic-robot-localization libpcl-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python related packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-setuptools \
    python3-cffi python3-numpy python3-dev \
    python3-catkin-pkg-modules && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install cython wheel && pip3 install \ 
    loguru stackprinter rospkg \
    catkin_pkg PyYAML && \
    rm -rf /root/.cache/pip/*

# Downlaod necessary repositories
RUN mkdir -p /catkin_ws/src && cd /catkin_ws/src \ 
    && git clone https://github.com/ANYbotics/kindr.git \
    && git clone https://github.com/ANYbotics/kindr_ros.git \
    && git clone https://github.com/anybotics/elevation_mapping.git \
    && cd elevation_mapping \
    && git checkout 04f28af523ed90b74d747df64417cedefec60f17 \
    && cd - \
    && mkdir SAFARI

RUN apt-get update && rm /etc/ros/rosdep/sources.list.d/20-default.list && rosdep init && rosdep update && \
    rosdep install --from-paths /catkin_ws/src/ --ignore-src --rosdistro melodic -r -y && \
    rm -rf /var/lib/apt/lists/*

# Build big packages
RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; cd /catkin_ws; catkin_make -DCMAKE_BUILD_TYPE=Release'

WORKDIR /catkin_ws
COPY . /catkin_ws/src/SAFARI

RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; cd /catkin_ws; catkin_make -DCMAKE_BUILD_TYPE=Release'

RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc && \
    echo "source /catkin_ws/devel/setup.bash" >> ~/.bashrc
