---
sidebar_position: 3
---

# Hardware Requirements

## Overview

This guide outlines the hardware requirements for implementing and experimenting with the Physical AI & Humanoid Robotics concepts covered in this textbook. The requirements vary depending on whether you're working with simulation, edge deployment, or full-scale humanoid robotics.

## System Requirements Overview

### Minimum Requirements
- **CPU**: `Intel i5 10th Gen` or `AMD Ryzen 5` equivalent  
- **GPU**: `NVIDIA GTX 1660` or equivalent with `6GB+ VRAM`  
- **RAM**: `32 GB DDR4`  
- **Storage**: `500 GB SSD`  
- **OS**: `Ubuntu 22.04 LTS` or `Windows 10/11` with WSL2  

### Recommended Requirements
- **CPU**: `Intel i9 13th Gen` or `AMD Ryzen 9 7000 series`  
- **GPU**: `NVIDIA RTX 4070 Ti` or `RTX 3090/4090` (`24GB+ VRAM`)  
- **RAM**: `64 GB DDR5`  
- **Storage**: `1 TB+ NVMe SSD`  
- **Network**: `Gigabit Ethernet`, `802.11ac WiFi`  

## Simulation Rig Requirements

### GPU Requirements for Simulation
- **Minimum**: `NVIDIA RTX 3060` (`12GB VRAM`)  
- **Recommended**: `NVIDIA RTX 4080/4090` (`24GB+ VRAM`)  
- **VRAM**: `12GB+ for complex scenes`  
- **CUDA**: `CUDA 11.8 or later required`  

### CPU Requirements for Physics Simulation
- **Cores**: `8+ physical cores recommended`  
- **Threads**: `16+ threads for complex multi-robot simulations`  
- **Architecture**: `Modern architecture with good single-core performance`  

### Memory Requirements
- **System RAM**: `32GB minimum`, `64GB recommended for large environments`  
- **VRAM**: `8GB minimum`, `24GB+ recommended for Isaac Sim`  

## Edge Kit Requirements

### NVIDIA Jetson Platforms

#### Jetson Orin Nano
- **GPU**: `1024-core NVIDIA Ampere architecture GPU`  
- **CPU**: `6-core ARM Cortex-A78AE v8.2 64-bit`  
- **Memory**: `4GB/8GB LPDDR5`  
- **Storage**: `16GB eMMC 5.1`  
- **Connectivity**: `Gigabit Ethernet`, `M.2 Key-E slot for Wi-Fi/Bluetooth`  
- **Power**: `15W-25W` consumption  
- **Use Case**: `Light perception and control tasks`  

#### Jetson Orin NX
- **GPU**: `1024-core NVIDIA Ampere architecture GPU`  
- **CPU**: `8-core ARM Cortex-A78AE v8.2 64-bit`  
- **Memory**: `8GB/16GB LPDDR5`  
- **Storage**: `16GB eMMC 5.1`  
- **Connectivity**: `Dual Gigabit Ethernet`, `M.2 Key-E`  
- **Power**: `25W-40W` consumption  
- **Use Case**: `Moderate perception and navigation`  

### Sensor Integration Requirements

#### Intel RealSense D435i
- **Stereo cameras**: `1280x720 at 90 FPS, 1280x720 at 30 FPS`  
- **IMU**: `Accelerometer and gyroscope`  
- **Connectivity**: `USB 3.0 Type-C`  
- **Range**: `0.22m to 9.7m`
- **FOV**: `85 deg x 58 deg x 94 deg`

#### Intel RealSense D455
- **Stereo cameras**: `1280x720 at 90 FPS, 1920x1080 at 30 FPS`
- **IMU**: `Accelerometer and gyroscope`
- **Connectivity**: `USB 3.0 Type-C`
- **Range**: `0.1m to 11m`
- **FOV**: `86 deg x 54 deg x 98 deg`
- **Advanced features**: `Embedded processing, multiple operation modes`  

#### USB IMU (BNO055)
- **Sensors**: `Accelerometer, gyroscope, magnetometer`  
- **Connectivity**: `USB interface`  
- **Fusion**: `On-board sensor fusion`  
- **Accuracy**: `<2° heading accuracy`  
- **Update Rate**: `Up to 100 Hz`  

#### Microphone/Speaker Array
- **Microphones**: `4+ microphones for beamforming`  
- **Sample Rate**: `16kHz or higher for speech recognition`  
- **Connectivity**: `USB or I2S interface`  
- **Features**: `Noise cancellation, echo suppression`  

## Robot Lab Options

### Proxy Robots (Quadruped/Robotic Arm)

#### Unitree Go2
- **Type**: `Quadruped robot`  
- **Actuators**: `12 joint motors (3 per leg)`  
- **Sensors**: `IMU, depth camera, stereo camera`  
- **Processing**: `On-board computing capability`  
- **Connectivity**: `WiFi, Ethernet`  
- **Battery**: `2.5 hours operation time`  
- **Payload**: `1kg`  
- **Use Case**: `Locomotion research, navigation`  

#### Popular Robotic Arms
- **Franka Emika Panda**: `7 DOF, torque control, 3kg payload`  
- **UR3/UR5/UR10**: `Universal Robots series, various payloads`  
- **Kinova Gen3**: `7 DOF, torque control, vision capabilities`  

### Miniature Humanoid Options

#### Unitree G1
- **Height**: `110cm`  
- **Weight**: `35kg`  
- **DOF**: `32`  
- **Actuators**: `High-torque servo motors`  
- **Sensors**: `Multiple IMUs, cameras, force/torque sensors`  
- **Computing**: `On-board computer`  
- **Battery**: `2+ hours operation`  
- **Payload**: `1.5kg in each hand`  

#### Robotis OP3
- **Height**: `75cm`  
- **Weight**: `9.5kg`  
- **DOF**: `28`  
- **Actuators**: `DYNAMIXEL PRO servos`  
- **Sensors**: `RGB-D camera, IMU, force sensors`  
- **Computing**: `Intel NUC or equivalent`  
- **Battery**: `1+ hours operation`  

#### Hiwonder TonyPi
- **Height**: `52cm`  
- **Weight**: `2.2kg`  
- **DOF**: `32`  
- **Actuators**: `High-precision servos`  
- **Sensors**: `Camera, microphone array`  
- **Computing**: `Raspberry Pi 4 or equivalent`  
- **Battery**: `1.5+ hours operation`  

## Cloud Option Requirements

### AWS EC2 Instance Recommendations

#### Graphics-Optimized Instances
- **g5.xlarge**: `1x A10G GPU, 4 vCPUs, 16GB memory`  
- **g5.2xlarge**: `1x A10G GPU, 8 vCPUs, 32GB memory`  
- **g5.4xlarge**: `1x A10G GPU, 16 vCPUs, 64GB memory`  
- **g5.8xlarge**: `1x A10G GPU, 32 vCPUs, 128GB memory`  
- **g5.16xlarge**: `1x A10G GPU, 64 vCPUs, 256GB memory`  

#### High Memory Instances for Simulation
- **g5.12xlarge**: `4x A10G GPUs, 48 vCPUs, 192GB memory`  
- **g5.24xlarge**: `4x A10G GPUs, 96 vCPUs, 384GB memory`  

### Azure GPU Recommendations
- **ND A100 v4 Series**: `NVIDIA A100 80GB GPUs`  
- **NCas T4 v3 Series**: `NVIDIA T4 GPUs`  
- **NVv4 Series**: `AMD Radeon Pro V600 GPUs`  

### Google Cloud Platform
- **A2 Instance Family**: `NVIDIA A100 GPUs`  
- **G2 Instance Family**: `NVIDIA L4 GPUs`  

## Network Requirements

### Local Network
- **Speed**: `Gigabit Ethernet recommended`  
- **Latency**: `less than 1ms for real-time control`  
- **Bandwidth**: `100 Mbps minimum for sensor data`  

### Wireless Requirements
- **WiFi 6 (802.11ax)**: `For high-bandwidth sensor data`  
- **5GHz Band**: `Less congestion than 2.4GHz`  
- **Access Points**: `Enterprise-grade for multiple robots`  

## Power Requirements

### UPS Systems
- **Capacity**: `15-30 minute runtime for safe shutdown`  
- **Power**: `1500VA+ for simulation workstation`  
- **Outlets**: `Surge-protected outlets for equipment`  

### Mobile Robot Power
- **Voltage**: `12V-48V depending on platform`  
- **Capacity**: `50Ah+ for extended operation`  
- **Charging**: `Automated charging stations`  

## Development Workstation Setup

### Multi-Monitor Configuration
- **Primary**: `27" 4K monitor for development`  
- **Secondary**: `24" 1080p for simulation/visualization`  
- **Tertiary**: `24" 1080p for documentation/command line`  

### Peripherals
- **Keyboard**: `Mechanical keyboard for extended coding`  
- **Mouse**: `High-DPI mouse with programmable buttons`  
- **Tablet**: `Graphics tablet for drawing diagrams (optional)`  

### Cooling Requirements
- **Case Fans**: `Adequate airflow for GPU cooling`  
- **Room Temperature**: `less than 25 degrees C for optimal performance`  
- **Acoustic**: `Consider noise levels in shared spaces`  

## Budget Considerations

### Entry Level (~$2000-5000)
- Development workstation: `Mid-range GPU (RTX 3060/3070)`  
- Simulation: `Basic Gazebo, limited Isaac Sim`  
- Hardware: `Used robot platforms, basic sensors`  
- Cloud: `Limited cloud compute time`  

### Professional Level (~$10,000-25,000)
- Development workstation: `High-end GPU (RTX 4080/4090)`  
- Simulation: `Full Isaac Sim capabilities`  
- Hardware: `New robot platforms, comprehensive sensors`  
- Cloud: `Moderate cloud compute resources`  

### Research Level (~$50,000+)
- Development cluster: `Multiple high-end workstations`  
- Simulation: `Multiple Isaac Sim instances`  
- Hardware: `Premium humanoid robots, complete sensor suites`  
- Cloud: `Extensive cloud compute resources`  

## Environmental Requirements

### Temperature
- Operating: `15-30 degrees C for electronics`  
- Storage: `-10 to 60 degrees C for most components`  
- Humidity: `20-80% non-condensing`  

### Space Requirements
- Workstation: `Dedicated desk space (2m x 1m minimum)`  
- Robot Testing: `Open area for robot movement (5m x 5m minimum)`  
- Storage: `Secure storage for equipment and components`  

### Safety Considerations
- Emergency Stop: `Easily accessible emergency stop for robots`  
- Barriers: `Physical barriers during testing`  
- Ventilation: `Adequate ventilation for electronics`  
- Fire Safety: `Appropriate fire suppression for electronics`  

## Maintenance and Upgrades

### Regular Maintenance
- Cleaning: `Dust cleaning for GPU and CPU coolers monthly`  
- Updates: `Regular OS and driver updates`  
- Backup: `Regular system backups of configurations`  

### Upgrade Path
- GPU: `Most important upgrade for simulation performance`  
- RAM: `Easy upgrade path for memory-intensive tasks`  
- Storage: `Consider additional SSDs for project isolation`  

## Vendor Recommendations

### Robot Platforms
- Unitree: `Advanced quadruped and humanoid robots`  
- Robotis: `Educational and research humanoid platforms`  
- Hiwonder: `Affordable humanoid robots for education`  
- Boston Dynamics: `Premium robot platforms (Spot, Atlas)`  

### Sensors
- Intel RealSense: `High-quality depth cameras`  
- Velodyne: `Professional LiDAR systems`  
- FLIR: `Thermal imaging cameras`  
- SICK: `Industrial LiDAR and vision systems`  

### Computing Platforms
- NVIDIA Jetson: `Edge AI computing`  
- Intel NUC: `Compact high-performance computing`  
- Raspberry Pi: `Low-cost computing for simple tasks`  

## Getting Started Checklist

### Before Purchase
- Define primary use case (`simulation vs. real hardware`)  
- Set budget and prioritize components  
- Verify software compatibility  
- Consider power and space requirements  
- Plan for future upgrades  

### Initial Setup
- Verify all components work independently  
- Install required software stack  
- Test basic functionality  
- Create backup configurations  
- Document system specifications  

### Safety Verification
- Test emergency stop procedures  
- Verify safe operating procedures  
- Ensure proper ventilation  
- Check electrical safety  

Your hardware selection should align with your specific goals, budget, and available space. Start with the minimum requirements and upgrade components as needed based on your actual usage patterns.
