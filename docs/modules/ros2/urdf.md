---
sidebar_position: 3
---

# URDF for Humanoid Robots

## Overview

URDF (Unified Robot Description Format) is an XML format used in ROS to describe robot models. It defines the physical and visual properties of a robot, including its links, joints, and other components. For humanoid robots, URDF is essential for simulation, visualization, and kinematic analysis.

## URDF Structure

A basic URDF file consists of:
- **Links**: Rigid bodies of the robot
- **Joints**: Connections between links
- **Visual**: How the robot appears in simulation
- **Collision**: Collision properties for physics simulation
- **Inertial**: Mass, center of mass, and inertia properties

## Basic URDF Example

```xml
<?xml version="1.0"?>
<robot name="simple_robot">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Child link connected via joint -->
  <joint name="base_to_top" type="fixed">
    <parent link="base_link"/>
    <child link="top_link"/>
    <origin xyz="0 0 0.3"/>
  </joint>

  <link name="top_link">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
      <material name="red">
        <color rgba="0.8 0 0 1"/>
      </material>
    </visual>
  </link>
</robot>
```

## Links

Links represent rigid bodies in the robot. Each link can have:

### Visual Properties
- **Geometry**: Shape (box, cylinder, sphere, mesh)
- **Material**: Color and texture
- **Origin**: Position and orientation relative to parent

### Collision Properties
- **Geometry**: Same as visual but for collision detection
- **Origin**: Position and orientation for collision

### Inertial Properties
- **Mass**: Mass of the link
- **Inertia**: Inertia tensor values
- **Origin**: Center of mass location

## Joints

Joints connect links and define how they can move relative to each other:

### Joint Types
- **fixed**: No movement between links
- **revolute**: Rotational movement around an axis (with limits)
- **continuous**: Continuous rotation around an axis (no limits)
- **prismatic**: Linear sliding movement along an axis
- **floating**: 6-DOF movement (for base of mobile robots)
- **planar**: Movement in a plane

### Joint Example
```xml
<joint name="shoulder_joint" type="revolute">
  <parent link="torso"/>
  <child link="upper_arm"/>
  <origin xyz="0.0 0.2 0.5" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  <dynamics damping="0.1" friction="0.0"/>
</joint>
```

## Humanoid Robot URDF

For humanoid robots, the URDF typically includes:

### Body Structure
- Torso
- Head
- Left/Right arms (with shoulder, elbow, wrist joints)
- Left/Right legs (with hip, knee, ankle joints)

### Example Humanoid Fragment
```xml
<?xml version="1.0"?>
<robot name="humanoid_robot">
  <!-- Torso -->
  <link name="torso">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Head -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.3"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="10" velocity="1"/>
  </joint>

  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="skin">
        <color rgba="1 0.8 0.6 1"/>
      </material>
    </visual>
  </link>

  <!-- Left Arm -->
  <joint name="left_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.2 0.1 0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="1"/>
  </joint>

  <link name="left_upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
      <material name="gray"/>
    </visual>
  </link>
</robot>
```

## Xacro for Complex Models

For complex humanoid robots, Xacro (XML Macros) is often used to simplify URDF:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_with_xacro">
  <!-- Define properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />

  <!-- Define a macro for a simple arm -->
  <xacro:macro name="simple_arm" params="prefix parent *origin">
    <joint name="${prefix}_shoulder_joint" type="revolute">
      <parent link="${parent}"/>
      <child link="${prefix}_upper_arm"/>
      <xacro:insert_block name="origin"/>
      <axis xyz="0 1 0"/>
      <limit lower="-1.57" upper="1.57" effort="50" velocity="1"/>
    </joint>

    <link name="${prefix}_upper_arm">
      <visual>
        <geometry>
          <cylinder length="0.3" radius="0.05"/>
        </geometry>
        <material name="gray">
          <color rgba="0.5 0.5 0.5 1"/>
        </material>
      </visual>
    </link>
  </xacro:macro>

  <!-- Use the macro -->
  <link name="torso">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
    </visual>
  </link>

  <xacro:simple_arm prefix="left" parent="torso">
    <origin xyz="0.2 0.1 0.2" rpy="0 0 0"/>
  </xacro:simple_arm>

  <xacro:simple_arm prefix="right" parent="torso">
    <origin xyz="-0.2 0.1 0.2" rpy="0 0 0"/>
  </xacro:simple_arm>
</robot>
```

## Working with URDF in ROS 2

### Launching Robot State Publisher
```bash
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:='$(find-pkg-share my_robot_description)/urdf/robot.urdf'
```

### Visualizing in RViz2
```bash
ros2 run rviz2 rviz2
# Add RobotModel display and set topic to /robot_description
```

### Using with Gazebo
```xml
<gazebo>
  <plugin name="joint_state_publisher" filename="libgazebo_ros_joint_state_publisher.so">
    <joint_name>joint_name</joint_name>
  </plugin>
</gazebo>
```

## Best Practices

1. **Coordinate Frames**: Follow ROS conventions (X forward, Y left, Z up)
2. **Units**: Use meters for distances, radians for angles
3. **Mass and Inertia**: Provide realistic values for proper physics simulation
4. **Joint Limits**: Always specify appropriate limits for revolute joints
5. **Meshes**: Use efficient meshes for visualization, simpler shapes for collision
6. **Xacro**: Use Xacro for complex robots to reduce redundancy

## Practical Exercise

Create a simple humanoid robot URDF with:
1. A torso and head
2. Two arms with shoulder and elbow joints
3. Two legs with hip and knee joints
4. Proper visual and collision properties
5. Use Xacro to avoid repetition

## Summary

- URDF describes robot structure in XML format
- Links define rigid bodies, joints define connections
- Visual, collision, and inertial properties define robot characteristics
- Xacro simplifies complex robot descriptions
- Proper URDF is essential for simulation and visualization
- Follow ROS conventions for coordinate frames and units