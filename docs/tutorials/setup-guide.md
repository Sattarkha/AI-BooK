---
sidebar_position: 1
---

# Setup Guide

## Overview

This guide provides comprehensive instructions for setting up the development environment for the Physical AI & Humanoid Robotics textbook. The setup includes ROS 2, simulation environments (Gazebo, Isaac Sim, Unity), NVIDIA Isaac packages, and the necessary tools for Vision-Language-Action development.

## Prerequisites

### System Requirements
- **Operating System**: Ubuntu 22.04 LTS (recommended) or Windows 10/11 with WSL2
- **CPU**: Intel i7 13th Gen+ or AMD Ryzen 9 (minimum), Intel i5 10th Gen+ or AMD Ryzen 5 (minimum acceptable)
- **GPU**: NVIDIA RTX 4070 Ti+ (recommended), RTX 3080+ (minimum for Isaac Sim), any GPU with CUDA support (for basic development)
- **RAM**: 64 GB DDR5 (recommended), 32 GB (minimum acceptable)
- **Storage**: 500 GB SSD (recommended), 250 GB (minimum)

### Software Prerequisites
- Git
- Python 3.10 or 3.11
- Docker (recommended for Isaac Sim)
- NVIDIA drivers (for GPU-accelerated features)
- CUDA Toolkit 11.8 or later

## ROS 2 Humble Hawksbill Installation

### 1. Set up sources
```bash
# Add ROS 2 GPG key and repository
sudo apt update && sudo apt install -y curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

### 2. Install ROS 2 packages
```bash
sudo apt update
sudo apt install ros-humble-desktop-full
sudo apt install python3-colcon-common-extensions
sudo apt install python3-rosdep
```

### 3. Initialize rosdep
```bash
sudo rosdep init
rosdep update
```

### 4. Source ROS 2 environment
```bash
# Add to ~/.bashrc for permanent sourcing
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## Simulation Environment Setup

### Gazebo Garden Installation
```bash
# Install Gazebo Garden
sudo apt install ros-humble-gazebo-*

# For standalone Gazebo
sudo apt install gz-garden

# Install Gazebo ROS packages
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control
```

### Isaac Sim Setup

#### Option 1: Docker Installation (Recommended)
```bash
# Pull Isaac Sim Docker image
docker pull nvcr.io/nvidia/isaac-sim:4.0.0

# Create a script to run Isaac Sim
cat << 'EOF' > ~/run_isaac_sim.sh
#!/bin/bash
xhost +local:docker
docker run --gpus all \
  --network=host \
  --env="DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --volume="${PWD}:/workspace" \
  --volume="/home/$USER/.Xauthority:/root/.Xauthority" \
  --runtime=nvidia \
  --privileged \
  --name isaac-sim \
  nvcr.io/nvidia/isaac-sim:4.0.0
EOF

chmod +x ~/run_isaac_sim.sh
```

#### Option 2: Standalone Installation
```bash
# Download Isaac Sim from NVIDIA Developer website
# Follow the installation guide at https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html
```

### Unity Setup for Robotics

#### 1. Install Unity Hub
```bash
# Download Unity Hub from unity3d.com
# Install Unity 2022.3 LTS version
```

#### 2. Install Unity Robotics packages
- Open Unity Hub
- Install Unity 2022.3 LTS
- Create a new 3D project
- In Package Manager, install:
  - Unity Robotics Hub
  - ROS-TCP-Connector
  - Unity Perception

## NVIDIA Isaac Packages Installation

### Isaac ROS Common
```bash
# Create workspace
mkdir -p ~/isaac_ros_ws/src
cd ~/isaac_ros_ws

# Clone Isaac ROS common repository
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git

# Install dependencies
rosdep install --from-paths src --ignore-src -r -y

# Build the workspace
colcon build --symlink-install
source install/setup.bash
```

### Isaac ROS Navigation and Perception
```bash
# Clone navigation packages
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam.git src/isaac_ros_visual_slam
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_manipulation.git src/isaac_ros_manipulation
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_apriltag.git src/isaac_ros_apriltag
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference.git src/isaac_ros_dnn_inference

# Install dependencies and build
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

## Development Tools Setup

### Python Environment
```bash
# Create Python virtual environment
python3 -m venv ~/physical_ai_env
source ~/physical_ai_env/bin/activate

# Install required Python packages
pip install --upgrade pip
pip install numpy scipy matplotlib opencv-python
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install openai spacy transformers
pip install open3d open3d-ml
pip install speechrecognition pyaudio webrtcvad
pip install networkx
```

### Install spaCy English model
```bash
python -m spacy download en_core_web_sm
```

### Node.js and Docusaurus for Textbook
```bash
# Install Node.js (version 18 or higher)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Install Docusaurus globally
npm install -g @docusaurus/cli

# Install textbook dependencies
cd /path/to/textbook/repository
npm install
```

## Hardware-Specific Setup

### NVIDIA Jetson Setup (Edge Kit)
If using NVIDIA Jetson for edge deployment:

```bash
# Flash Jetson with appropriate image
# Install JetPack SDK
# Install ROS 2 Humble on Jetson
sudo apt install ros-humble-desktop-full

# Install Isaac ROS for Jetson
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git
# Follow Jetson-specific build instructions
```

### Sensor Integration
```bash
# For Intel RealSense D435i/D455
sudo apt install ros-humble-realsense2-camera

# For USB IMU (BNO055)
pip install adafruit-circuitpython-bno055

# For microphone/speaker array
sudo apt install alsa-utils pulseaudio
```

## Verification Steps

### 1. Verify ROS 2 Installation
```bash
# Check ROS 2 installation
ros2 --version

# Run a simple ROS 2 demo
source /opt/ros/humble/setup.bash
ros2 run demo_nodes_cpp talker
```

### 2. Verify Gazebo Installation
```bash
# Launch Gazebo
gz sim

# Or with ROS 2 integration
ros2 launch gazebo_ros gazebo.launch.py
```

### 3. Verify Isaac Sim (if installed)
```bash
# Run Isaac Sim Docker container
~/run_isaac_sim.sh

# Or check if Isaac Sim is properly installed
# Launch Isaac Sim and verify the interface appears
```

### 4. Test Basic ROS 2 Commands
```bash
# Create a test workspace
mkdir -p ~/test_ws/src
cd ~/test_ws

# Build and source
colcon build
source install/setup.bash

# Verify communication
ros2 topic list
```

## Common Setup Issues and Solutions

### 1. CUDA/GPU Issues
```bash
# Check NVIDIA driver and CUDA installation
nvidia-smi
nvcc --version

# If CUDA is not detected, reinstall drivers:
sudo apt purge nvidia-*
sudo apt autoremove
# Download and install from NVIDIA website
```

### 2. ROS 2 Permission Issues
```bash
# Fix ROS 2 permission issues
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
```

### 3. Gazebo Plugin Issues
```bash
# If Gazebo plugins don't load:
echo 'export GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH:/usr/lib/x86_64-linux-gnu/gazebo-11/plugins' >> ~/.bashrc
echo 'export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/usr/share/gazebo-11/models' >> ~/.bashrc
source ~/.bashrc
```

### 4. Isaac Sim Docker Issues
```bash
# Check Docker installation
docker --version

# Ensure Docker has GPU access
sudo usermod -aG docker $USER
# Log out and back in

# Test GPU access in Docker
docker run --gpus all nvidia/cuda:11.0-base-ubuntu20.04 nvidia-smi
```

## Environment Configuration

### Create Environment Setup Script
```bash
cat << 'EOF' > ~/setup_physical_ai_env.sh
#!/bin/bash

# Source ROS 2
source /opt/ros/humble/setup.bash

# Source Isaac ROS workspace if built
if [ -f ~/isaac_ros_ws/install/setup.bash ]; then
    source ~/isaac_ros_ws/install/setup.bash
fi

# Set up Python environment
source ~/physical_ai_env/bin/activate

# Set Isaac Sim path if installed
export ISAAC_SIM_PATH="$HOME/.local/share/ov/pkg/isaac_sim-4.0.0"

# Set Gazebo paths
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/.gazebo/models
export GAZEBO_RESOURCE_PATH=$GAZEBO_RESOURCE_PATH:~/.gazebo

echo "Physical AI environment configured!"
EOF

chmod +x ~/setup_physical_ai_env.sh
```

## Testing the Complete Setup

### 1. Launch a Basic Simulation
```bash
# Source the environment
source ~/setup_physical_ai_env.sh

# Launch Gazebo with an empty world
ros2 launch gazebo_ros empty_world.launch.py
```

### 2. Test ROS 2 Communication
```bash
# In one terminal
ros2 run demo_nodes_cpp talker

# In another terminal
ros2 run demo_nodes_cpp listener
```

### 3. Test Isaac ROS Package (if installed)
```bash
# Test Visual SLAM
ros2 launch isaac_ros_visual_slam isaac_ros_visual_slam_stereo.launch.py
```

## Next Steps

After completing the setup:

1. **Run the tutorials**: Start with basic ROS 2 tutorials
2. **Explore simulation**: Try the Gazebo and Isaac Sim environments
3. **Test perception**: Run basic perception demos
4. **Read the textbook**: Navigate through the modules in order
5. **Try examples**: Execute the code examples provided in each module

## Troubleshooting Resources

- **ROS 2 Documentation**: https://docs.ros.org/en/humble/
- **Isaac Sim Documentation**: https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html
- **Isaac ROS Documentation**: https://nvidia-isaac-ros.github.io/
- **Gazebo Documentation**: https://gazebosim.org/docs/
- **Community Support**: ROS Answers, NVIDIA Developer Forums

## Updating the Environment

### Update ROS 2 Packages
```bash
sudo apt update
sudo apt upgrade ros-humble-*
```

### Update Isaac ROS Packages
```bash
cd ~/isaac_ros_ws
git pull
colcon build --symlink-install
```

### Update Python Packages
```bash
source ~/physical_ai_env/bin/activate
pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 pip install -U
```

Your development environment is now ready for working with the Physical AI & Humanoid Robotics textbook content. Proceed to the next tutorial to start with basic ROS 2 concepts.