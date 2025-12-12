---
sidebar_position: 3
---

# Nav2 Path Planning

## Overview

Navigation2 (Nav2) is the state-of-the-art navigation stack for ROS 2, providing advanced path planning, localization, and navigation capabilities. It's particularly important for humanoid robots that require sophisticated navigation in complex environments with dynamic obstacles.

## Key Components of Nav2

Nav2 consists of several key components that work together to provide complete navigation functionality:

- **Navigation Lifecycle**: Manages the navigation system state
- **Global Planner**: Computes optimal path from start to goal
- **Local Planner**: Executes path while avoiding obstacles
- **Controller**: Low-level control for robot motion
- **Recovery Behaviors**: Actions when navigation fails
- **Sensors**: Integration with various sensor types

## Architecture

### Navigation System States
```
IDLE → CLEANING → SPINNING → WAITING → MOVING
  ↑                                      ↓
  ←------------- RECOVERING ←-------------↓
```

### Main Components
- **Navigation Server**: Coordinates all navigation components
- **Map Server**: Provides static map information
- **Local/Global Costmap**: Represents obstacles and free space
- **Transform System**: Maintains coordinate frames (TF)

## Installation and Setup

### Installing Nav2
```bash
# Install Nav2 packages
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup

# Or build from source
git clone https://github.com/ros-planning/navigation2.git
cd navigation2
rosdep install -i --from-path src --rosdistro humble -y
colcon build --packages-select nav2_bringup
```

### Basic Launch
```bash
# Launch navigation with default configuration
ros2 launch nav2_bringup navigation_launch.py

# Launch with simulation
ros2 launch nav2_bringup tb3_simulation_launch.py
```

## Costmap Configuration

Costmaps represent the environment as a 2D grid with cost values indicating how difficult it is to traverse each cell.

### Global Costmap
```yaml
# config/global_costmap.yaml
global_costmap:
  global_frame: map
  robot_base_frame: base_link
  update_frequency: 1.0
  publish_frequency: 1.0
  static_map: true
  rolling_window: false
  resolution: 0.05  # meters per cell
  inflation_radius: 0.55

  plugins:
    - {name: static_layer, type: "nav2_costmap_2d::StaticLayer"}
    - {name: obstacle_layer, type: "nav2_costmap_2d::ObstacleLayer"}
    - {name: inflation_layer, type: "nav2_costmap_2d::InflationLayer"}
```

### Local Costmap
```yaml
# config/local_costmap.yaml
local_costmap:
  global_frame: odom
  robot_base_frame: base_link
  update_frequency: 5.0
  publish_frequency: 2.0
  static_map: false
  rolling_window: true
  width: 3
  height: 3
  resolution: 0.05

  plugins:
    - {name: obstacle_layer, type: "nav2_costmap_2d::ObstacleLayer"}
    - {name: voxel_layer, type: "nav2_costmap_2d::VoxelLayer"}
    - {name: inflation_layer, type: "nav2_costmap_2d::InflationLayer"}
```

## Global Path Planning

### Available Global Planners
- **NavFn**: Fast marching method for global planning
- **Global Planner**: A* implementation
- **TEB Planner**: Timed Elastic Band for trajectory optimization
- **SMAC Planner**: Sparse Markov Chain for SE2/3 planning

### Configuring Global Planner
```yaml
# config/planner_server.yaml
planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner/NavfnPlanner"
      tolerance: 0.5
      use_astar: false
      allow_unknown: true
```

### Custom Path Planning Example
```python
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from nav2_msgs.srv import ComputePathToPose
import math

class CustomPathPlanner(Node):
    def __init__(self):
        super().__init__('custom_path_planner')

        # Service client for path computation
        self.path_client = self.create_client(
            ComputePathToPose,
            'compute_path_to_pose'
        )

        # Publisher for path visualization
        self.path_publisher = self.create_publisher(
            Path,
            'custom_path',
            10
        )

    def compute_path(self, start_pose, goal_pose):
        """Compute path from start to goal"""
        while not self.path_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Path service not available...')

        request = ComputePathToPose.Request()
        request.goal = goal_pose
        request.start = start_pose

        future = self.path_client.call_async(request)
        return future

    def smooth_path(self, path):
        """Apply path smoothing algorithm"""
        if len(path.poses) < 3:
            return path

        smoothed_path = Path()
        smoothed_path.header = path.header

        # Apply simple smoothing (Chaikin's algorithm)
        for i in range(len(path.poses) - 1):
            # Calculate intermediate points
            start_point = path.poses[i].pose.position
            end_point = path.poses[i + 1].pose.position

            # Create two new points between original points
            new_pose1 = PoseStamped()
            new_pose1.header = path.header
            new_pose1.pose.position.x = 0.75 * start_point.x + 0.25 * end_point.x
            new_pose1.pose.position.y = 0.75 * start_point.y + 0.25 * end_point.y
            new_pose1.pose.position.z = 0.75 * start_point.z + 0.25 * end_point.z

            new_pose2 = PoseStamped()
            new_pose2.header = path.header
            new_pose2.pose.position.x = 0.25 * start_point.x + 0.75 * end_point.x
            new_pose2.pose.position.y = 0.25 * start_point.y + 0.75 * end_point.y
            new_pose2.pose.position.z = 0.25 * start_point.z + 0.75 * end_point.z

            smoothed_path.poses.append(new_pose1)
            smoothed_path.poses.append(new_pose2)

        return smoothed_path

def main(args=None):
    rclpy.init(args=args)
    planner = CustomPathPlanner()

    # Example usage
    goal_pose = PoseStamped()
    goal_pose.header.frame_id = 'map'
    goal_pose.pose.position.x = 5.0
    goal_pose.pose.position.y = 5.0
    goal_pose.pose.position.z = 0.0
    goal_pose.pose.orientation.w = 1.0

    future = planner.compute_path(PoseStamped(), goal_pose)
    rclpy.spin_until_future_complete(planner, future)

    if future.result() is not None:
        path = future.result().path
        smoothed_path = planner.smooth_path(path)
        planner.path_publisher.publish(smoothed_path)
    else:
        planner.get_logger().error('Path computation failed')

    planner.destroy_node()
    rclpy.shutdown()
```

## Local Path Planning and Control

### Local Planner Configuration
```yaml
# config/controller_server.yaml
controller_server:
  ros__parameters:
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    controller_plugins: ["FollowPath"]

    FollowPath:
      plugin: "nav2_mppi_controller::MPPIController"
      time_steps: 50
      model_dt: 0.05
      vx_std: 0.2
      vy_std: 0.1
      wz_std: 0.4
      vx_max: 0.5
      vx_min: -0.15
      vy_max: 1.0
      wz_max: 1.0
      simulation_time: 1.6
      speed_scaling_factor: 0.2
      control_cost_factor: 0.0
      goal_cost_factor: 2.0
      path_cost_factor: 2.0
      occ_cost_factor: 2.0
      heading_cost_factor: 0.1
      oscillation_cost_factor: 0.5
```

### Alternative: DWB Controller
```yaml
# Using DWB (Dynamic Window Approach) controller
controller_server:
  ros__parameters:
    controller_plugins: ["FollowPath"]

    FollowPath:
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
```

## Navigation for Humanoid Robots

### Special Considerations for Humanoids
Humanoid robots have unique navigation challenges:

1. **Bipedal Dynamics**: More complex than wheeled robots
2. **Stability**: Must maintain balance during movement
3. **Foot Placement**: Requires careful footstep planning
4. **Upper Body**: Arms and head affect center of mass

### Footstep Planning
```python
import numpy as np
from geometry_msgs.msg import Point

class FootstepPlanner:
    def __init__(self):
        self.step_length = 0.3  # meters
        self.step_width = 0.2   # meters (lateral distance between feet)
        self.step_height = 0.05 # meters (step clearance)

    def plan_footsteps(self, path, robot_pose):
        """Plan footstep locations along a path"""
        footsteps = []

        # Convert path to footstep sequence
        for i in range(0, len(path.poses), 5):  # Sample every 5 poses
            pose = path.poses[i].pose

            # Calculate foot positions based on robot's current stance
            left_foot = Point()
            right_foot = Point()

            # For each step, calculate left and right foot positions
            # This is a simplified example - real implementation would be more complex
            if i % 10 == 0:  # Alternate feet
                left_foot.x = pose.position.x + self.step_width / 2
                left_foot.y = pose.position.y
                left_foot.z = 0.0  # Ground level

                right_foot.x = pose.position.x - self.step_width / 2
                right_foot.y = pose.position.y
                right_foot.z = 0.0
            else:
                right_foot.x = pose.position.x + self.step_width / 2
                right_foot.y = pose.position.y
                right_foot.z = 0.0

                left_foot.x = pose.position.x - self.step_width / 2
                left_foot.y = pose.position.y
                left_foot.z = 0.0

            footsteps.append(('left', left_foot))
            footsteps.append(('right', right_foot))

        return footsteps
```

### Stability-Aware Navigation
```python
class StabilityAwareNavigator:
    def __init__(self):
        self.zmp_margin = 0.05  # Zero Moment Point safety margin
        self.com_height = 0.8   # Center of mass height

    def is_path_stable(self, path, robot_state):
        """Check if path is stable for humanoid robot"""
        # Calculate ZMP (Zero Moment Point) along path
        for pose in path.poses:
            # Calculate center of mass projection
            com_proj = self.project_com(pose.pose.position)

            # Check if within support polygon
            if not self.is_in_support_polygon(com_proj, robot_state):
                return False  # Path is unstable

        return True

    def project_com(self, position):
        """Project center of mass to ground plane"""
        # Simplified projection - real implementation would consider full kinematics
        return (position.x, position.y)

    def is_in_support_polygon(self, com_proj, robot_state):
        """Check if COM projection is within support polygon"""
        # Calculate support polygon based on foot positions
        # This is a simplified version
        left_foot = robot_state.left_foot_pose.position
        right_foot = robot_state.right_foot_pose.position

        # Create convex hull of support polygon
        # Check if COM projection is inside
        return True  # Simplified
```

## Recovery Behaviors

### Available Recovery Behaviors
- **Spin**: Rotate in place to clear local minima
- **Backup**: Move backward to escape tight spaces
- **Wait**: Pause navigation temporarily

### Recovery Configuration
```yaml
# config/recovery_server.yaml
recovery_server:
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
```

## Behavior Trees for Navigation

### Overview
Nav2 uses behavior trees to define navigation logic:

```xml
<!-- example_behavior_tree.xml -->
<root main_tree_to_execute="MainTree">
  <BehaviorTree ID="MainTree">
    <PipelineSequence name="NavigateWithRecovery">
      <RecoveryNode number_of_retries="2" name="SpinRecovery">
        <Spin spin_dist="1.57"/>
        <Wait wait_duration="5"/>
      </RecoveryNode>
      <RecoveryNode number_of_retries="2" name="BackupRecovery">
        <BackUp backup_dist="0.15" backup_speed="0.025"/>
        <Wait wait_duration="5"/>
      </RecoveryNode>
      <PipelineSequence name="Navigate">
        <ComputePathToPose goal="{goal_pose}"/>
        <FollowPath path="{path}" controller_id="FollowPath"/>
      </PipelineSequence>
    </PipelineSequence>
  </BehaviorTree>
</root>
```

## Practical Implementation

### Creating a Navigation Node
```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose

class Nav2Navigator(Node):
    def __init__(self):
        super().__init__('nav2_navigator')
        self.nav_to_pose_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose'
        )

    def navigate_to_pose(self, pose):
        """Send navigation goal to Nav2"""
        # Wait for action server
        while not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('Waiting for navigation server...')

        # Create goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        # Send goal
        self.nav_to_pose_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )

    def feedback_callback(self, feedback_msg):
        """Handle navigation feedback"""
        self.get_logger().info(
            f'Navigation feedback: {feedback_msg.feedback.distance_remaining:.2f}m remaining'
        )

def main(args=None):
    rclpy.init(args=args)
    navigator = Nav2Navigator()

    # Create goal pose
    goal_pose = PoseStamped()
    goal_pose.header.frame_id = 'map'
    goal_pose.pose.position.x = 5.0
    goal_pose.pose.position.y = 5.0
    goal_pose.pose.orientation.w = 1.0

    # Start navigation
    navigator.navigate_to_pose(goal_pose)

    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        pass
    finally:
        navigator.destroy_node()
        rclpy.shutdown()
```

## Simulation and Testing

### Testing in Gazebo
```bash
# Launch Gazebo with TurtleBot3
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

# Launch navigation
ros2 launch nav2_bringup navigation_launch.py use_sim_time:=True

# Launch RViz for visualization
ros2 launch nav2_bringup view_navigation_launch.py use_sim_time:=True
```

### Automated Testing
```python
import unittest
from nav2_simple_commander.robot_navigator import BasicNavigator

class TestNavigation(unittest.TestCase):
    def setUp(self):
        self.navigator = BasicNavigator()

    def test_simple_navigation(self):
        """Test basic navigation to a goal"""
        goal_pose = self.create_pose(5.0, 5.0, 0.0)

        self.navigator.goToPose(goal_pose)

        # Wait for completion
        while not self.navigator.isTaskComplete():
            pass

        result = self.navigator.getResult()
        self.assertEqual(result, TaskResult.SUCCEEDED)

    def create_pose(self, x, y, theta):
        """Create a pose for testing"""
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.z = math.sin(theta / 2.0)
        pose.pose.orientation.w = math.cos(theta / 2.0)
        return pose
```

## Best Practices

1. **Parameter Tuning**: Carefully tune costmap and planner parameters for your robot
2. **Sensor Integration**: Ensure proper sensor data for obstacle detection
3. **Safety Margins**: Set appropriate inflation radii for safe navigation
4. **Recovery Behaviors**: Configure appropriate recovery behaviors for your environment
5. **Testing**: Extensively test navigation in simulation before real-world deployment
6. **Humanoid-Specific**: Consider balance and stability for humanoid robots

## Troubleshooting

### Common Issues
- **Path Not Found**: Check map quality and costmap parameters
- **Oscillation**: Adjust controller parameters and safety margins
- **Performance**: Optimize costmap resolution and update frequencies
- **Localization**: Ensure proper AMCL or SLAM setup

## Practical Exercise

Implement a navigation system for a humanoid robot:
1. Configure Nav2 for your robot platform
2. Set up costmaps with appropriate parameters
3. Implement path planning with stability considerations
4. Test navigation in simulation
5. Add custom recovery behaviors if needed

## Summary

- Nav2 provides comprehensive navigation capabilities for ROS 2
- Proper configuration of costmaps and planners is essential
- Humanoid robots require special considerations for balance and stability
- Recovery behaviors help handle navigation failures
- Extensive testing in simulation is recommended before deployment
- Behavior trees provide flexible navigation logic