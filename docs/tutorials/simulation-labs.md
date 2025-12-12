---
sidebar_position: 2
---

# Simulation Labs

## Overview

This section provides hands-on simulation laboratories that allow you to practice and experiment with the concepts covered in the Physical AI & Humanoid Robotics textbook. Each lab includes step-by-step instructions, expected outcomes, and troubleshooting tips.

## Lab 1: ROS 2 Basics - Nodes, Topics, and Services

### Objective
Learn the fundamental concepts of ROS 2 communication by creating and running simple nodes that publish and subscribe to topics, and provide services.

### Prerequisites
- Completed ROS 2 installation
- Basic understanding of ROS 2 concepts from the textbook

### Lab Steps

#### 1. Create a new ROS 2 package
```bash
# Source ROS 2
source /opt/ros/humble/setup.bash

# Create workspace if not already done
mkdir -p ~/ros2_labs/src
cd ~/ros2_labs

# Create a new package
ros2 pkg create --build-type ament_python ros2_basics_lab --dependencies rclpy std_msgs geometry_msgs
```

#### 2. Create a publisher node
Create the file `~/ros2_labs/src/ros2_basics_lab/ros2_basics_lab/publisher_member_function.py`:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

#### 3. Create a subscriber node
Create the file `~/ros2_labs/src/ros2_basics_lab/ros2_basics_lab/subscriber_member_function.py`:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

#### 4. Create a service server
Create the file `~/ros2_labs/src/ros2_basics_lab/ros2_basics_lab/service_member_function.py`:

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node


class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Returning: {response.sum}')
        return response


def main(args=None):
    rclpy.init(args=args)

    minimal_service = MinimalService()

    rclpy.spin(minimal_service)

    # Destroy the node explicitly
    minimal_service.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

#### 5. Create a service client
Create the file `~/ros2_labs/src/ros2_basics_lab/ros2_basics_lab/client_member_function.py`:

```python
from example_interfaces.srv import AddTwoInts
import sys
import rclpy
from rclpy.node import Node


class MinimalClientAsync(Node):

    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()


def main():
    rclpy.init()

    minimal_client = MinimalClientAsync()
    response = minimal_client.send_request(int(sys.argv[1]), int(sys.argv[2]))
    minimal_client.get_logger().info(
        f'Result of add_two_ints: for {sys.argv[1]} + {sys.argv[2]} = {response.sum}')

    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

#### 6. Update setup.py
Update the `~/ros2_labs/src/ros2_basics_lab/setup.py` file:

```python
from setuptools import find_packages, setup

package_name = 'ros2_basics_lab'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = ros2_basics_lab.publisher_member_function:main',
            'listener = ros2_basics_lab.subscriber_member_function:main',
            'server = ros2_basics_lab.service_member_function:main',
            'client = ros2_basics_lab.client_member_function:main',
        ],
    },
)
```

#### 7. Build and run the lab
```bash
cd ~/ros2_labs
colcon build --packages-select ros2_basics_lab
source install/setup.bash

# Terminal 1: Run the publisher
ros2 run ros2_basics_lab talker

# Terminal 2: Run the subscriber
ros2 run ros2_basics_lab listener

# Terminal 3: Run the service server
ros2 run ros2_basics_lab server

# Terminal 4: Run the client (in a new terminal)
ros2 run ros2_basics_lab client 1 2
```

### Expected Outcome
- Publisher node sends "Hello World: X" messages every 0.5 seconds
- Subscriber node receives and prints the messages
- Service server responds to addition requests
- Client receives and prints the sum of two numbers

### Troubleshooting
- If nodes don't communicate, check that ROS_DOMAIN_ID is the same across terminals
- If build fails, ensure all dependencies are installed

## Lab 2: Gazebo Simulation - Robot Modeling and Control

### Objective
Create a simple robot model in URDF, spawn it in Gazebo, and control its movement using ROS 2.

### Lab Steps

#### 1. Create a robot description package
```bash
cd ~/ros2_labs/src
ros2 pkg create --build-type ament_cmake robot_modeling_lab --dependencies urdf xacro
```

#### 2. Create a URDF model
Create the file `~/ros2_labs/src/robot_modeling_lab/urdf/simple_robot.urdf.xacro`:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="simple_robot">
  <!-- Properties -->
  <xacro:property name="base_width" value="0.5"/>
  <xacro:property name="base_length" value="0.8"/>
  <xacro:property name="base_height" value="0.2"/>
  <xacro:property name="wheel_radius" value="0.1"/>
  <xacro:property name="wheel_width" value="0.05"/>
  <xacro:property name="wheel_offset_x" value="0.2"/>
  <xacro:property name="wheel_offset_y" value="0.3"/>
  <xacro:property name="wheel_offset_z" value="0.0"/>

  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="${base_length} ${base_width} ${base_height}"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="${base_length} ${base_width} ${base_height}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Wheels -->
  <xacro:macro name="wheel" params="prefix x_reflect y_reflect">
    <link name="${prefix}_wheel">
      <visual>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
        <material name="black">
          <color rgba="0 0 0 1"/>
        </material>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="1"/>
        <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
      </inertial>
    </link>

    <joint name="${prefix}_wheel_joint" type="continuous">
      <parent link="base_link"/>
      <child link="${prefix}_wheel"/>
      <origin xyz="${x_reflect * wheel_offset_x} ${y_reflect * wheel_offset_y} ${wheel_offset_z - wheel_radius}" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
    </joint>
  </xacro:macro>

  <xacro:wheel prefix="front_left" x_reflect="1" y_reflect="1"/>
  <xacro:wheel prefix="front_right" x_reflect="1" y_reflect="-1"/>
  <xacro:wheel prefix="back_left" x_reflect="-1" y_reflect="1"/>
  <xacro:wheel prefix="back_right" x_reflect="-1" y_reflect="-1"/>
</robot>
```

#### 3. Create Gazebo world
Create the file `~/ros2_labs/src/robot_modeling_lab/worlds/simple_room.world`:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="simple_room">
    <!-- Include ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Include sun -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Room walls -->
    <model name="wall_1">
      <pose>0 5 1 0 0 0</pose>
      <link name="link">
        <pose>0 0 0 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.2 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.2 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
        <self_collide>0</self_collide>
        <enable_wind>0</enable_wind>
        <kinematic>0</kinematic>
      </link>
    </model>

    <model name="wall_2">
      <pose>5 0 1 0 0 1.5707</pose>
      <link name="link">
        <pose>0 0 0 0 0 0</pose>
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.2 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.2 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
        <self_collide>0</self_collide>
        <enable_wind>0</enable_wind>
        <kinematic>0</kinematic>
      </link>
    </model>
  </world>
</sdf>
```

#### 4. Create launch file
Create the file `~/ros2_labs/src/robot_modeling_lab/launch/robot_spawn.launch.py`:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare launch configuration
    urdf_model_path = PathJoinSubstitution([
        FindPackageShare('robot_modeling_lab'),
        'urdf',
        'simple_robot.urdf.xacro'
    ])

    world_path = PathJoinSubstitution([
        FindPackageShare('robot_modeling_lab'),
        'worlds',
        'simple_room.world'
    ])

    # Launch Gazebo with world
    gazebo = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'simple_robot',
            '-file', urdf_model_path,
            '-x', '0', '-y', '0', '-z', '0.5'
        ],
        output='screen'
    )

    # Launch Gazebo server and client
    gzserver = Node(
        package='gazebo_ros',
        executable='gzserver',
        arguments=[world_path],
        output='screen'
    )

    gzclient = Node(
        package='gazebo_ros',
        executable='gzclient',
        output='screen'
    )

    return LaunchDescription([
        gzserver,
        gzclient,
        gazebo
    ])
```

#### 5. Build and run the simulation
```bash
cd ~/ros2_labs
colcon build --packages-select robot_modeling_lab
source install/setup.bash

# Launch the simulation
ros2 launch robot_modeling_lab robot_spawn.launch.py
```

### Expected Outcome
- A simple robot model with 4 wheels appears in Gazebo
- The robot is placed in a room with walls
- You can visualize the robot in Gazebo

### Troubleshooting
- If the robot doesn't appear, check the URDF syntax
- If Gazebo doesn't start, ensure proper graphics drivers are installed

## Lab 3: Isaac Sim - Photorealistic Simulation

### Objective
Learn to use Isaac Sim for photorealistic simulation and synthetic data generation.

### Prerequisites
- Isaac Sim installed (Docker or standalone)

### Lab Steps

#### 1. Create Isaac Sim Python script
Create the file `~/isaac_sim_labs/basic_scene.py`:

```python
# basic_scene.py
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.prims import get_prim_at_path
import numpy as np

# Create a simple scene
def create_basic_scene():
    # Initialize the world
    world = World(stage_units_in_meters=1.0)

    # Add default ground plane
    world.scene.add_default_ground_plane()

    # Add a simple robot (if Isaac Sim assets are available)
    assets_root_path = get_assets_root_path()
    if assets_root_path:
        # Add a simple robot or object
        cube = world.scene.add(
            DynamicCuboid(
                prim_path="/World/Cube",
                name="cube",
                position=np.array([0.5, 0.5, 0.5]),
                size=np.array([0.2, 0.2, 0.2]),
                color=np.array([0.5, 0.0, 0.0])
            )
        )

    # Add a table
    table = world.scene.add(
        DynamicCuboid(
            prim_path="/World/Table",
            name="table",
            position=np.array([1.0, 0.0, 0.1]),
            size=np.array([0.8, 1.0, 0.2]),
            color=np.array([0.2, 0.2, 0.6])
        )
    )

    # Reset the world
    world.reset()

    # Run simulation for a few steps
    for i in range(100):
        world.step(render=True)
        if i % 20 == 0:
            print(f"Simulation step: {i}")

    # Clean up
    world.clear()
    print("Basic Isaac Sim scene completed!")

if __name__ == "__main__":
    create_basic_scene()
```

#### 2. Run the Isaac Sim lab
```bash
# If using Docker
docker run --gpus all --rm -it \
  --network=host \
  --env="DISPLAY" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --volume="$HOME/isaac_sim_labs:/workspace/isaac_sim_labs" \
  --name isaac_sim_lab \
  nvcr.io/nvidia/isaac-sim:4.0.0

# Inside the container:
cd /workspace/isaac_sim_labs
python basic_scene.py
```

### Expected Outcome
- Isaac Sim opens with a basic scene
- A cube and table are added to the scene
- Physics simulation runs for 100 steps

## Lab 4: Navigation with Nav2

### Objective
Implement autonomous navigation using the Navigation2 stack.

### Lab Steps

#### 1. Create navigation configuration
Create the file `~/ros2_labs/src/navigation_lab/config/nav2_params.yaml`:

```yaml
amcl:
  ros__parameters:
    use_sim_time: True
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_footprint"
    beam_span: 0.0628318530718
    bmt_threshold: 0.25
    do_beamskip: false
    do_reframe: true
    first_map_only: false
    global_frame_id: "map"
    initial_cov_xx: 0.25
    initial_cov_yy: 0.25
    initial_cov_aa: 0.0685389192651
    lambda_short: 0.05
    max_beams: 60
    max_particles: 2000
    min_particles: 500
    odom_frame_id: "odom"
    pf_err: 0.05
    pf_z: 0.99
    recovery_alpha_fast: 0.0
    recovery_alpha_slow: 0.0
    resample_interval: 1
    save_pose_rate: 0.5
    set_initial_pose: true
    sigma_hit: 0.2
    tf_broadcast: true
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.25
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05

bt_navigator:
  ros__parameters:
    use_sim_time: True
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    default_bt_xml_filename: "navigate_w_replanning_and_recovery.xml"
    plugin_lib_names:
    - nav2_compute_path_to_pose_action_bt_node
    - nav2_follow_path_action_bt_node
    - nav2_back_up_action_bt_node
    - nav2_spin_action_bt_node
    - nav2_wait_action_bt_node
    - nav2_clear_costmap_service_bt_node
    - nav2_is_stuck_condition_bt_node
    - nav2_goal_reached_condition_bt_node
    - nav2_goal_updated_condition_bt_node
    - nav2_initial_pose_received_condition_bt_node
    - nav2_reinitialize_global_localization_service_bt_node
    - nav2_rate_controller_bt_node
    - nav2_distance_controller_bt_node
    - nav2_speed_controller_bt_node
    - nav2_truncate_path_action_bt_node
    - nav2_goal_updater_node_bt_node
    - nav2_recovery_node_bt_node
    - nav2_pipeline_sequence_bt_node
    - nav2_round_robin_node_bt_node
    - nav2_transform_available_condition_bt_node
    - nav2_time_expired_condition_bt_node
    - nav2_path_expiring_timer_condition
    - nav2_distance_traveled_condition_bt_node
    - nav2_single_trigger_bt_node
    - nav2_is_battery_low_condition_bt_node
    - nav2_navigate_through_poses_action_bt_node
    - nav2_navigate_to_pose_action_bt_node
    - nav2_remove_passed_goals_action_bt_node
    - nav2_planner_selector_bt_node
    - nav2_controller_selector_bt_node
    - nav2_goal_checker_selector_bt_node

controller_server:
  ros__parameters:
    use_sim_time: True
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    # Progress checker parameters
    progress_checker:
      plugin: "nav2_controller::SimpleProgressChecker"
      required_movement_radius: 0.5
      movement_time_allowance: 10.0

    # Goal checker parameters
    goal_checker:
      plugin: "nav2_controller::SimpleGoalChecker"
      xy_goal_tolerance: 0.25
      yaw_goal_tolerance: 0.25
      stateful: True

    # Controller parameters
    FollowPath:
      plugin: "nav2_rotation_shim_controller::RotationShimController"
      primary_controller: "dwb_core::DWBLocalPlanner"

      # DWB parameters
      dwb:
        plugin: "dwb_core::DWBLocalPlanner"
        debug_trajectory_details: True
        min_vel_x: 0.0
        min_vel_y: 0.0
        max_vel_x: 0.5
        max_vel_y: 0.0
        max_vel_theta: 1.0
        min_speed_xy: 0.0
        max_speed_xy: 0.5
        min_speed_theta: 0.0
        acc_lim_x: 2.5
        acc_lim_y: 0.0
        acc_lim_theta: 3.2
        decel_lim_x: -2.5
        decel_lim_y: 0.0
        decel_lim_theta: -3.2
        vx_samples: 20
        vy_samples: 0
        vtheta_samples: 40
        sim_time: 1.7
        linear_granularity: 0.05
        angular_granularity: 0.025
        transform_tolerance: 0.2
        xy_goal_tolerance: 0.25
        trans_stopped_velocity: 0.25
        short_circuit_trajectory_evaluation: True
        stateful: True
        critics: ["RotateToGoal", "Oscillation", "BaseObstacle", "GoalAlign", "PathAlign", "PathDist", "GoalDist"]
        BaseObstacle.scale: 0.02
        PathAlign.scale: 32.0
        PathAlign.forward_point_distance: 0.1
        GoalAlign.scale: 24.0
        GoalAlign.forward_point_distance: 0.1
        PathDist.scale: 32.0
        GoalDist.scale: 24.0
        RotateToGoal.scale: 32.0
        RotateToGoal.slowing_factor: 5.0
        RotateToGoal.lookahead_time: -1.0

global_costmap:
  ros__parameters:
    update_frequency: 1.0
    publish_frequency: 1.0
    global_frame: map
    robot_base_frame: base_link
    use_sim_time: True
    rolling_window: false
    width: 20
    height: 20
    resolution: 0.05
    origin_x: -10.0
    origin_y: -10.0
    plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
    static_layer:
      plugin: "nav2_costmap_2d::StaticLayer"
      map_subscribe_transient_local: true
    obstacle_layer:
      plugin: "nav2_costmap_2d::ObstacleLayer"
      enabled: True
      observation_sources: scan
      scan:
        topic: /scan
        max_obstacle_height: 2.0
        clearing: True
        marking: True
        data_type: "LaserScan"
    inflation_layer:
      plugin: "nav2_costmap_2d::InflationLayer"
      cost_scaling_factor: 3.0
      inflation_radius: 0.55
    always_send_full_costmap: True

local_costmap:
  ros__parameters:
    update_frequency: 5.0
    publish_frequency: 2.0
    global_frame: odom
    robot_base_frame: base_link
    use_sim_time: True
    rolling_window: true
    width: 3
    height: 3
    resolution: 0.05
    origin_x: -1.5
    origin_y: -1.5
    plugins: ["obstacle_layer", "inflation_layer"]
    obstacle_layer:
      plugin: "nav2_costmap_2d::ObstacleLayer"
      enabled: True
      observation_sources: scan
      scan:
        topic: /scan
        max_obstacle_height: 2.0
        clearing: True
        marking: True
        data_type: "LaserScan"
    inflation_layer:
      plugin: "nav2_costmap_2d::InflationLayer"
      cost_scaling_factor: 3.0
      inflation_radius: 0.55
    always_send_full_costmap: True

planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    use_sim_time: True
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner/NavfnPlanner"
      tolerance: 0.5
      use_astar: false
      allow_unknown: true

recoveries_server:
  ros__parameters:
    costmap_topic: local_costmap/costmap_raw
    footprint_topic: local_costmap/published_footprint
    cycle_frequency: 10.0
    recovery_plugins: ["spin", "backup", "wait"]
    spin:
      plugin: "nav2_recoveries/Spin"
      spin_dist: 1.57
    backup:
      plugin: "nav2_recoveries/BackUp"
      backup_dist: 0.15
      backup_speed: 0.025
    wait:
      plugin: "nav2_recoveries/Wait"
      wait_duration: 1.0

waypoint_follower:
  ros__parameters:
    loop_rate: 20
    stop_on_failure: false
    waypoint_task_executor_plugin: "wait_at_waypoint"
    wait_at_waypoint:
      plugin: "nav2_waypoint_follower::WaitAtWaypoint"
      enabled: true
      wait_time: 1
```

#### 2. Create navigation launch file
Create the file `~/ros2_labs/src/navigation_lab/launch/navigation.launch.py`:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Launch configuration
    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')

    # Declare launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='True',
        description='Use simulation (Gazebo) clock if true'
    )

    declare_params_file = DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('navigation_lab'),
            'config',
            'nav2_params.yaml'
        ]),
        description='Full path to the navigation parameters file to use'
    )

    # Launch navigation
    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('nav2_bringup'),
                'launch',
                'navigation_launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'params_file': params_file
        }.items()
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_params_file,
        navigation_launch
    ])
```

#### 3. Run navigation simulation
```bash
# First terminal - launch Gazebo world
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

# Second terminal - launch navigation
ros2 launch nav2_bringup navigation_launch.py use_sim_time:=True

# Third terminal - launch RViz for visualization
ros2 launch nav2_bringup view_navigation_launch.py use_sim_time:=True
```

### Expected Outcome
- Navigation stack initializes successfully
- Robot can be commanded to navigate to different locations
- Costmaps update based on sensor data
- Robot avoids obstacles during navigation

## Lab 5: Vision-Language-Action Integration

### Objective
Create a complete VLA system that accepts voice commands, processes them, and executes robotic actions.

### Lab Steps

#### 1. Create VLA node
Create the file `~/ros2_labs/src/vla_lab/vla_lab/vla_node.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import speech_recognition as sr
import threading
import time
import openai
import json

class VLANode(Node):
    def __init__(self):
        super().__init__('vla_node')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.tts_pub = self.create_publisher(String, '/tts_input', 10)

        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Initialize components
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Adjust for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

        # Initialize OpenAI (set your API key)
        # openai.api_key = "your-openai-api-key"

        # Start voice recognition thread
        self.voice_thread = threading.Thread(target=self.voice_recognition_loop)
        self.voice_thread.daemon = True
        self.voice_thread.start()

        self.get_logger().info("VLA Node initialized")

    def scan_callback(self, msg):
        """Handle laser scan data"""
        # Process laser scan for obstacle detection
        min_distance = min(msg.ranges)
        if min_distance < 0.5:  # Obstacle within 0.5m
            self.get_logger().warn(f"Obstacle detected at {min_distance:.2f}m")

    def voice_recognition_loop(self):
        """Continuously listen for voice commands"""
        while rclpy.ok():
            try:
                with self.microphone as source:
                    self.get_logger().info("Listening for command...")
                    audio = self.recognizer.listen(source, timeout=5.0, phrase_time_limit=5.0)

                # Recognize speech
                command = self.recognizer.recognize_google(audio)
                self.get_logger().info(f"Heard command: {command}")

                # Process the command
                self.process_command(command.lower())

            except sr.WaitTimeoutError:
                # No speech detected, continue listening
                continue
            except sr.UnknownValueError:
                self.get_logger().warn("Could not understand audio")
            except sr.RequestError as e:
                self.get_logger().error(f"Speech recognition error: {e}")
            except Exception as e:
                self.get_logger().error(f"Error in voice recognition: {e}")

    def process_command(self, command):
        """Process voice command and execute appropriate action"""
        self.get_logger().info(f"Processing command: {command}")

        # Simple command parsing
        if "forward" in command or "go" in command and "forward" in command:
            self.move_robot(0.2, 0.0)  # Move forward at 0.2 m/s
        elif "backward" in command or "back" in command:
            self.move_robot(-0.2, 0.0)  # Move backward
        elif "left" in command:
            self.move_robot(0.0, 0.5)  # Turn left
        elif "right" in command:
            self.move_robot(0.0, -0.5)  # Turn right
        elif "stop" in command or "halt" in command:
            self.stop_robot()
        else:
            # For complex commands, use LLM (simplified here)
            self.handle_complex_command(command)

    def move_robot(self, linear_vel, angular_vel):
        """Move robot with specified velocities"""
        twist = Twist()
        twist.linear.x = linear_vel
        twist.angular.z = angular_vel

        self.cmd_vel_pub.publish(twist)
        self.get_logger().info(f"Moving robot - linear: {linear_vel}, angular: {angular_vel}")

    def stop_robot(self):
        """Stop robot movement"""
        twist = Twist()
        self.cmd_vel_pub.publish(twist)
        self.get_logger().info("Robot stopped")

    def handle_complex_command(self, command):
        """Handle complex commands using LLM (simplified implementation)"""
        # In a real implementation, this would call an LLM
        # to understand and decompose complex commands
        self.get_logger().info(f"Complex command received: {command}")

        # For now, just acknowledge the command
        response_msg = String()
        response_msg.data = f"I received your command: {command}. This is a simplified response."
        self.tts_pub.publish(response_msg)

def main(args=None):
    rclpy.init(args=args)
    vla_node = VLANode()

    try:
        rclpy.spin(vla_node)
    except KeyboardInterrupt:
        pass
    finally:
        vla_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### 2. Update setup.py for VLA package
Update the `~/ros2_labs/src/vla_lab/setup.py` file:

```python
from setuptools import find_packages, setup

package_name = 'vla_lab'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='VLA Lab Package',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vla_node = vla_lab.vla_node:main',
        ],
    },
)
```

#### 3. Build and run VLA lab
```bash
cd ~/ros2_labs
colcon build --packages-select vla_lab
source install/setup.bash

# Run the VLA node
ros2 run vla_lab vla_node
```

### Expected Outcome
- Node starts and begins listening for voice commands
- Simple commands (forward, backward, left, right, stop) control robot movement
- Complex commands are acknowledged

## Best Practices for Simulation Labs

1. **Start Simple**: Begin with basic examples before moving to complex scenarios
2. **Incremental Development**: Build functionality step by step
3. **Testing**: Verify each component before integration
4. **Documentation**: Keep notes on what works and what doesn't
5. **Safety**: Always test in simulation before real hardware
6. **Version Control**: Use Git to track changes and experiments

## Troubleshooting Common Issues

### Gazebo Issues
- **Black screen**: Check graphics drivers and OpenGL support
- **Slow performance**: Reduce model complexity or physics update rate
- **Model not spawning**: Verify URDF syntax and file paths

### ROS 2 Issues
- **Nodes not communicating**: Check ROS_DOMAIN_ID and network configuration
- **Build failures**: Verify dependencies and package.xml
- **Parameter errors**: Check parameter names and types

### Isaac Sim Issues
- **Docker permissions**: Add user to docker group
- **GPU not detected**: Verify NVIDIA drivers and CUDA installation
- **Performance issues**: Check system requirements and resources

## Next Steps

After completing these simulation labs:

1. **Experiment**: Modify the examples to understand how they work
2. **Combine concepts**: Integrate different modules together
3. **Advanced topics**: Move to more complex scenarios
4. **Real hardware**: Transition to physical robots when ready
5. **Capstone project**: Apply all learned concepts to complete project

These labs provide hands-on experience with the core concepts of Physical AI and Humanoid Robotics. Practice each lab multiple times to gain confidence in the tools and concepts.