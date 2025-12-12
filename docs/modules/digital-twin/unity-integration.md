---
sidebar_position: 2
---

# Unity Integration

## Overview

Unity is a powerful 3D development platform that can be used for creating high-fidelity simulation environments for robotics. While Gazebo excels at physics simulation, Unity provides superior graphics capabilities and a rich ecosystem for creating immersive and visually appealing digital twins of robotic systems.

## Unity for Robotics

Unity's robotics integration provides:
- **High-quality rendering**: Photorealistic graphics for visual perception training
- **XR support**: Virtual and augmented reality capabilities
- **Asset ecosystem**: Extensive library of 3D models and environments
- **Real-time simulation**: Interactive environments for robot development
- **Cross-platform deployment**: Deploy to various hardware platforms

## Unity Robotics Hub

The Unity Robotics Hub is a collection of tools and packages that facilitate robotics development:

### Key Components:
- **ROS-TCP-Connector**: Communication bridge between Unity and ROS/ROS 2
- **Unity Perception**: Tools for generating synthetic training data
- **Unity Robotics Simulation**: Framework for large-scale robot simulation

## Setting up Unity for Robotics

### Prerequisites:
1. Unity Hub and Unity Editor (2021.3 LTS or later recommended)
2. Unity Robotics Hub package
3. ROS/ROS 2 installation on your system
4. Python for ROS communication

### Installation Steps:
1. Install Unity Hub from unity3d.com
2. Install Unity Editor with Linux Build Support (if using ROS on Linux)
3. Create a new 3D project
4. Import the ROS-TCP-Connector package from Unity Asset Store or GitHub

## ROS-TCP-Connector

The ROS-TCP-Connector enables communication between Unity and ROS:

### Unity Side:
```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;

public class RobotController : MonoBehaviour
{
    ROSConnection ros;
    string rosIP = "127.0.0.1";
    int rosPort = 10000;

    void Start()
    {
        ros = ROSConnection.instance;
        ros.Initialize(rosIP, rosPort);
    }

    void Update()
    {
        // Send a message to ROS
        if (Input.GetKeyDown(KeyCode.Space))
        {
            var twist = new TwistMsg();
            twist.linear = new Vector3Msg(1, 0, 0); // Move forward
            ros.Send("cmd_vel", twist);
        }
    }

    // Receive messages from ROS
    void OnMessageReceived(MessageType msg)
    {
        // Handle received messages
    }
}
```

### ROS Side:
```python
import rospy
from geometry_msgs.msg import Twist
from unity_robotics_demo_msgs.msg import UnityTwist  # Custom message type

def unity_cmd_vel_callback(data):
    # Convert Unity message to ROS message
    cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
    twist = Twist()
    twist.linear.x = data.linear.x
    twist.linear.y = data.linear.y
    twist.linear.z = data.linear.z
    twist.angular.x = data.angular.x
    twist.angular.y = data.angular.y
    twist.angular.z = data.angular.z
    cmd_vel_pub.publish(twist)

def main():
    rospy.init_node('unity_bridge')
    rospy.Subscriber('unity_cmd_vel', UnityTwist, unity_cmd_vel_callback)
    rospy.spin()

if __name__ == '__main__':
    main()
```

## Unity Perception Package

The Unity Perception package enables synthetic data generation for AI training:

### Key Features:
- **Semantic segmentation**: Pixel-perfect labeling of objects
- **Bounding boxes**: 2D and 3D bounding box annotations
- **Depth images**: Ground truth depth information
- **Camera intrinsics**: Accurate camera parameters
- **Custom sensors**: Extend with custom sensor types

### Example: Adding Perception Camera
```csharp
using Unity.Robotics.Perception;
using UnityEngine;

public class PerceptionSetup : MonoBehaviour
{
    public void Start()
    {
        var sensor = gameObject.AddComponent<CameraSensor>();
        sensor.sensorName = "Main Camera";
        sensor.frequency = 10.0f; // Hz

        // Add annotation interfaces
        var segmentation = gameObject.AddComponent<SyntheticDataSegmentation>();
        var boundingBox = gameObject.AddComponent<SyntheticDataBoundingBox2D>();
    }
}
```

## Creating a Robot in Unity

### 1. Import Robot Model
- Import your robot model as a 3D asset
- Set up colliders for physics interactions
- Configure joint constraints for realistic movement

### 2. Robot Controller Script
```csharp
using UnityEngine;

public class URDFRobotController : MonoBehaviour
{
    public GameObject[] joints;
    public float[] jointPositions;
    public float[] jointVelocities;

    void Update()
    {
        // Update joint positions based on ROS commands
        for (int i = 0; i < joints.Length; i++)
        {
            if (i < jointPositions.Length)
            {
                // Apply joint position update
                joints[i].transform.localRotation =
                    Quaternion.Euler(0, jointPositions[i], 0);
            }
        }
    }

    public void SetJointPositions(float[] positions)
    {
        jointPositions = positions;
    }
}
```

## Unity Simulation Framework

For large-scale robot simulation:

### Orchestrator Pattern:
- **Coordinator**: Manages multiple simulation instances
- **Worker**: Individual simulation environments
- **Database**: Stores simulation results and metrics

### Example Architecture:
```
[Coordinator]
    |
    ├── [Worker 1] - Simulation Instance 1
    ├── [Worker 2] - Simulation Instance 2
    ├── [Worker 3] - Simulation Instance 3
    └── [Database] - Results Storage
```

## Sensor Simulation in Unity

### Camera Sensor:
```csharp
using Unity.Robotics.Perception;
using UnityEngine;

public class CameraSensor : MonoBehaviour
{
    public Camera camera;
    public float updateRate = 30.0f;

    void Start()
    {
        camera = GetComponent<Camera>();
        InvokeRepeating("CaptureFrame", 0, 1.0f/updateRate);
    }

    void CaptureFrame()
    {
        // Capture image and send to ROS
        // Process with perception pipeline
    }
}
```

### LiDAR Simulation:
Unity doesn't have built-in LiDAR, but you can simulate it using raycasting:

```csharp
using UnityEngine;

public class LiDARSimulation : MonoBehaviour
{
    public int resolution = 360;
    public float range = 10.0f;

    void Update()
    {
        float[] ranges = new float[resolution];

        for (int i = 0; i < resolution; i++)
        {
            float angle = (float)i / resolution * 360.0f;
            Vector3 direction = Quaternion.Euler(0, angle, 0) * transform.forward;

            if (Physics.Raycast(transform.position, direction, out RaycastHit hit, range))
            {
                ranges[i] = hit.distance;
            }
            else
            {
                ranges[i] = range; // No obstacle detected
            }
        }

        // Send ranges to ROS
    }
}
```

## Human-Robot Interaction in Unity

Unity excels at creating human-robot interaction scenarios:

### 1. Gesture Recognition
- Use Unity's input system for gesture capture
- Simulate human gestures for robot perception

### 2. Voice Integration
- Integrate with speech recognition APIs
- Simulate voice commands for robot processing

### 3. Social Navigation
- Create scenarios with human agents
- Test robot navigation in populated environments

## Performance Optimization

### Rendering Optimization:
- Use occlusion culling for large environments
- Implement Level of Detail (LOD) systems
- Optimize materials and lighting

### Physics Optimization:
- Use simplified collision meshes
- Adjust physics update rates based on requirements
- Implement object pooling for dynamic objects

## Best Practices

1. **Asset Quality**: Balance visual fidelity with performance
2. **Communication**: Use efficient message protocols between Unity and ROS
3. **Scalability**: Design for multiple simulation instances
4. **Realism**: Include realistic sensor noise and latency
5. **Testing**: Validate simulation results against real-world data
6. **Documentation**: Maintain clear documentation of Unity-ROS interfaces

## Practical Exercise

Create a Unity scene with:
1. A simple robot model with basic movement
2. ROS-TCP-Connector integration
3. Camera sensor publishing images to ROS
4. Basic navigation in a Unity environment
5. Human-robot interaction scenario

## Integration with ROS 2

For ROS 2 integration, use the ROS2Unity package:

```csharp
using ROS2;
using ROS2.Utils;

public class ROS2RobotController : MonoBehaviour
{
    ROS2UnityComponent ros2Component;

    void Start()
    {
        ros2Component = GetComponent<ROS2UnityComponent>();
        ros2Component.Subscribe<TwistMsg>("cmd_vel", HandleCmdVel);
    }

    void HandleCmdVel(TwistMsg msg)
    {
        // Process velocity command
        transform.Translate(new Vector3(msg.linear.x, 0, msg.linear.y) * Time.deltaTime);
    }
}
```

## Summary

- Unity provides high-fidelity graphics for robotics simulation
- ROS-TCP-Connector enables communication with ROS/ROS 2
- Unity Perception generates synthetic training data
- Unity excels at human-robot interaction scenarios
- Performance optimization is crucial for large-scale simulation
- Unity complements Gazebo by providing superior visualization