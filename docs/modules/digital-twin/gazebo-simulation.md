---
sidebar_position: 1
---

# Gazebo Simulation

## Overview

Gazebo is a powerful 3D simulation environment that provides accurate physics simulation, high-quality graphics, and convenient programmatic interfaces. It's widely used in robotics for testing algorithms, training robots, and validating designs before deployment on real hardware.

## Key Features of Gazebo

- **Physics Simulation**: Accurate simulation of rigid body dynamics, collisions, and contacts
- **Sensor Simulation**: Support for various sensors including cameras, LiDAR, IMUs, GPS, etc.
- **3D Visualization**: High-quality rendering with realistic lighting and materials
- **Plugin System**: Extensible architecture for custom sensors and controllers
- **ROS Integration**: Seamless integration with ROS/ROS 2 for robot simulation

## Installing Gazebo

Gazebo comes with ROS installations, but you can also install it standalone:

```bash
# For Ubuntu with ROS 2 Humble
sudo apt install ros-humble-gazebo-*

# Or install Gazebo Garden (standalone)
sudo apt install gazebo
```

## Basic Gazebo Concepts

### Worlds
A world file (.world) defines the environment, including:
- Physics properties
- Models and their initial positions
- Lighting and environment settings

Example world file:
```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="default">
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Add your robot model here -->
    <include>
      <uri>model://my_robot</uri>
      <pose>0 0 0.5 0 0 0</pose>
    </include>

    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>
  </world>
</sdf>
```

### Models
Models (.sdf or .urdf) represent robots and objects in the simulation:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="simple_box">
    <link name="box_link">
      <pose>0 0 0.5 0 0 0</pose>
      <collision name="box_collision">
        <geometry>
          <box>
            <size>1 1 1</size>
          </box>
        </geometry>
      </collision>
      <visual name="box_visual">
        <geometry>
          <box>
            <size>1 1 1</size>
          </box>
        </geometry>
        <material>
          <ambient>1 0 0 1</ambient>
          <diffuse>1 0 0 1</diffuse>
        </material>
      </visual>
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.1667</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.1667</iyy>
          <iyz>0.0</iyz>
          <izz>0.1667</izz>
        </inertia>
      </inertial>
    </link>
  </model>
</sdf>
```

## Creating Your First Simulation

### 1. Launch Gazebo
```bash
# Launch Gazebo with an empty world
gazebo

# Or launch with a specific world file
gazebo my_world.world
```

### 2. Spawn a Robot
```bash
# If you have a URDF model
ros2 run gazebo_ros spawn_entity.py -entity my_robot -file /path/to/robot.urdf -x 0 -y 0 -z 1
```

## Physics Simulation

Gazebo supports multiple physics engines:
- **ODE (Open Dynamics Engine)**: Default, good for most applications
- **Bullet**: More stable for complex multi-body systems
- **DART**: Advanced physics with better contact handling

### Physics Parameters
- **Max Step Size**: Simulation time step (typically 0.001s)
- **Real Time Factor**: Simulation speed relative to real time
- **Real Time Update Rate**: Updates per second

## Sensor Simulation

Gazebo provides realistic simulation of various sensors:

### Camera Sensor
```xml
<sensor name="camera" type="camera">
  <camera>
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
    </image>
    <clip>
      <near>0.1</near>
      <far>100</far>
    </clip>
  </camera>
  <update_rate>30</update_rate>
  <always_on>true</always_on>
</sensor>
```

### LiDAR Sensor
```xml
<sensor name="lidar" type="ray">
  <ray>
    <scan>
      <horizontal>
        <samples>360</samples>
        <resolution>1.0</resolution>
        <min_angle>-3.14159</min_angle>
        <max_angle>3.14159</max_angle>
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>
      <max>30.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
  <update_rate>10</update_rate>
</sensor>
```

### IMU Sensor
```xml
<sensor name="imu" type="imu">
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
        </noise>
      </z>
    </angular_velocity>
  </imu>
</sensor>
```

## Gazebo Plugins

Plugins extend Gazebo's functionality:

### Joint Control Plugin
```xml
<plugin name="joint_state_publisher" filename="libgazebo_ros_joint_state_publisher.so">
  <joint_name>joint_name</joint_name>
</plugin>
```

### Diff Drive Plugin (for wheeled robots)
```xml
<plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
  <left_joint>left_wheel_joint</left_joint>
  <right_joint>right_wheel_joint</right_joint>
  <wheel_separation>0.4</wheel_separation>
  <wheel_diameter>0.2</wheel_diameter>
  <command_topic>cmd_vel</command_topic>
  <odometry_topic>odom</odometry_topic>
</plugin>
```

## ROS 2 Integration

### Launching with ROS 2
```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('gazebo_ros'),
                    'launch',
                    'gazebo.launch.py'
                ])
            ]),
            launch_arguments={'world': PathJoinSubstitution([
                FindPackageShare('my_robot_description'),
                'worlds',
                'my_world.world'
            ])}
        )
    ])
```

### Controlling the Robot
```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        msg = Twist()
        msg.linear.x = 1.0  # Move forward
        msg.angular.z = 0.5  # Turn
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    controller = RobotController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()
```

## Best Practices

1. **Model Quality**: Use appropriate mesh complexity for simulation speed vs. visual quality
2. **Physics Tuning**: Adjust physics parameters based on simulation requirements
3. **Sensor Noise**: Add realistic noise models to sensor data
4. **Collision Detection**: Use simple collision geometries for better performance
5. **World Design**: Create worlds that match your testing requirements
6. **Plugin Selection**: Use appropriate plugins for your robot's functionality

## Debugging Tips

- Use Gazebo's GUI to visualize collision meshes and joint axes
- Check TF frames in RViz to ensure proper robot state publishing
- Monitor simulation timing to maintain real-time performance
- Use Gazebo's built-in tools to inspect model properties and physics

## Practical Exercise

Create a simple mobile robot simulation:
1. Design a basic robot with two wheels and a chassis
2. Add a camera sensor to the robot
3. Create a world with obstacles
4. Implement basic movement control
5. Test the robot's navigation in the simulated environment

## Summary

- Gazebo provides realistic physics simulation for robotics
- Worlds define the environment and physics properties
- Models represent robots and objects in the simulation
- Sensors provide realistic data for robot algorithms
- Plugins extend functionality for specific robot types
- ROS 2 integration enables seamless robot simulation and control