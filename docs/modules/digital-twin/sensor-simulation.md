---
sidebar_position: 3
---

# Sensor Simulation

## Overview

Sensor simulation is a critical component of digital twin environments, enabling robots to perceive their virtual surroundings in ways that closely mimic real-world sensors. Accurate sensor simulation is essential for developing, testing, and validating robotic perception algorithms before deployment on physical hardware.

## Types of Sensors in Robotics

### 1. Range Sensors
- **LiDAR**: Light Detection and Ranging - provides 2D or 3D point clouds
- **Ultrasonic**: Measures distance using sound waves
- **Infrared**: Short-range distance measurement

### 2. Vision Sensors
- **RGB Cameras**: Color image capture
- **Stereo Cameras**: Depth estimation through disparity
- **RGB-D Cameras**: Color + depth information (e.g., Kinect, RealSense)

### 3. Inertial Sensors
- **IMU**: Inertial Measurement Unit - measures acceleration and angular velocity
- **Gyroscope**: Measures angular velocity
- **Accelerometer**: Measures linear acceleration

### 4. Other Sensors
- **GPS**: Global Positioning System
- **Magnetometer**: Measures magnetic field (compass)
- **Force/Torque Sensors**: Measures applied forces and torques

## LiDAR Simulation

### Principles
LiDAR sensors emit laser beams and measure the time it takes for the light to return after reflecting off objects.

### Simulation Parameters
- **Range**: Maximum and minimum detection distance
- **Resolution**: Angular resolution of the sensor
- **Field of View**: Horizontal and vertical FOV
- **Noise**: Simulated measurement uncertainty

### Gazebo LiDAR Simulation
```xml
<sensor name="lidar" type="ray">
  <ray>
    <scan>
      <horizontal>
        <samples>720</samples>
        <resolution>1</resolution>
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
  <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
    <ros>
      <namespace>lidar</namespace>
      <remapping>~/out:=scan</remapping>
    </ros>
    <output_type>sensor_msgs/LaserScan</output_type>
  </plugin>
</sensor>
```

### Unity LiDAR Simulation (Raycasting Approach)
```csharp
using UnityEngine;
using System.Collections.Generic;

public class LiDARSimulation : MonoBehaviour
{
    [Header("LiDAR Parameters")]
    public int horizontalRays = 360;
    public int verticalRays = 1;
    public float maxRange = 20.0f;
    public float minRange = 0.1f;
    public LayerMask detectionMask = -1;

    [Header("Noise Parameters")]
    public float noiseStdDev = 0.02f;

    private List<float> ranges;

    void Start()
    {
        ranges = new List<float>(new float[horizontalRays * verticalRays]);
    }

    void Update()
    {
        SimulateLiDAR();
    }

    void SimulateLiDAR()
    {
        for (int h = 0; h < horizontalRays; h++)
        {
            float hAngle = (float)h / horizontalRays * 360.0f;

            for (int v = 0; v < verticalRays; v++)
            {
                float vAngle = (float)v / verticalRays * 10.0f - 5.0f; // Small vertical spread

                Vector3 direction = Quaternion.Euler(vAngle, hAngle, 0) * transform.forward;

                if (Physics.Raycast(transform.position, direction, out RaycastHit hit, maxRange, detectionMask))
                {
                    float distance = hit.distance;
                    // Add noise to simulate real sensor
                    distance += RandomGaussian() * noiseStdDev;

                    int index = h * verticalRays + v;
                    ranges[index] = Mathf.Clamp(distance, minRange, maxRange);
                }
                else
                {
                    int index = h * verticalRays + v;
                    ranges[index] = maxRange;
                }
            }
        }

        // Publish ranges to ROS or process locally
        PublishRanges();
    }

    float RandomGaussian()
    {
        // Box-Muller transform for Gaussian noise
        float u1 = Random.value;
        float u2 = Random.value;
        return Mathf.Sqrt(-2.0f * Mathf.Log(u1)) * Mathf.Cos(2.0f * Mathf.PI * u2);
    }

    void PublishRanges()
    {
        // Publish to ROS topic or Unity event system
    }
}
```

## Camera Simulation

### RGB Camera
Simulates a standard color camera:

#### Gazebo Camera Configuration
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
  <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
    <ros>
      <namespace>camera</namespace>
      <remapping>~/image_raw:=image</remapping>
    </ros>
  </plugin>
</sensor>
```

### RGB-D Camera Simulation
Combines color and depth information:

#### Unity RGB-D Simulation
```csharp
using UnityEngine;

[RequireComponent(typeof(Camera))]
public class RGBDCamera : MonoBehaviour
{
    [Header("Camera Parameters")]
    public int width = 640;
    public int height = 480;
    public float maxDepth = 10.0f;
    public float minDepth = 0.1f;

    private Camera cam;
    private RenderTexture depthTexture;
    private Texture2D rgbTexture;

    void Start()
    {
        cam = GetComponent<Camera>();

        // Create textures for RGB and depth
        rgbTexture = new Texture2D(width, height, TextureFormat.RGB24, false);

        depthTexture = new RenderTexture(width, height, 24);
        depthTexture.format = RenderTextureFormat.Depth;
        cam.targetTexture = depthTexture;
    }

    void Update()
    {
        // Capture RGB image
        RenderTexture.active = null;
        rgbTexture.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        rgbTexture.Apply();

        // Process depth information
        ProcessDepth();
    }

    void ProcessDepth()
    {
        // Extract depth information from camera's depth buffer
        // Convert to depth map for robotics applications
    }
}
```

## IMU Simulation

### Principles
IMUs measure linear acceleration and angular velocity, often including magnetometer data for orientation.

#### Gazebo IMU Configuration
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
          <bias_mean>0.0000075</bias_mean>
          <bias_stddev>0.0000008</bias_stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
          <bias_mean>0.0000075</bias_mean>
          <bias_stddev>0.0000008</bias_stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
          <bias_mean>0.0000075</bias_mean>
          <bias_stddev>0.0000008</bias_stddev>
        </noise>
      </z>
    </angular_velocity>
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.1</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.1</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </node>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.1</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
</sensor>
```

### Unity IMU Simulation
```csharp
using UnityEngine;

public class IMUSimulation : MonoBehaviour
{
    [Header("IMU Parameters")]
    public float accelerometerNoise = 0.01f;
    public float gyroscopeNoise = 0.001f;
    public float magnetometerNoise = 0.1f;

    private Vector3 lastPosition;
    private Quaternion lastRotation;
    private float lastTime;

    void Start()
    {
        lastPosition = transform.position;
        lastRotation = transform.rotation;
        lastTime = Time.time;
    }

    void Update()
    {
        float deltaTime = Time.time - lastTime;

        if (deltaTime > 0)
        {
            // Calculate linear acceleration
            Vector3 velocity = (transform.position - lastPosition) / deltaTime;
            Vector3 lastVelocity = (lastPosition - transform.position) / deltaTime; // Previous velocity
            Vector3 linearAcceleration = (velocity - lastVelocity) / deltaTime;

            // Add gravity compensation
            linearAcceleration -= Physics.gravity;

            // Add noise
            linearAcceleration += Random.insideUnitSphere * accelerometerNoise;

            // Calculate angular velocity
            Quaternion deltaRotation = transform.rotation * Quaternion.Inverse(lastRotation);
            Vector3 angularVelocity = new Vector3(
                Mathf.Atan2(2 * (deltaRotation.x * deltaRotation.w - deltaRotation.y * deltaRotation.z),
                           1 - 2 * (deltaRotation.x * deltaRotation.x + deltaRotation.z * deltaRotation.z)) / deltaTime,
                Mathf.Atan2(2 * (deltaRotation.y * deltaRotation.w + deltaRotation.x * deltaRotation.z),
                           1 - 2 * (deltaRotation.y * deltaRotation.y + deltaRotation.z * deltaRotation.z)) / deltaTime,
                Mathf.Atan2(2 * (deltaRotation.z * deltaRotation.w - deltaRotation.x * deltaRotation.y),
                           1 - 2 * (deltaRotation.x * deltaRotation.x + deltaRotation.y * deltaRotation.y)) / deltaTime
            );

            // Add noise
            angularVelocity += Random.insideUnitSphere * gyroscopeNoise;

            // Publish IMU data
            PublishIMUData(linearAcceleration, angularVelocity);
        }

        lastPosition = transform.position;
        lastRotation = transform.rotation;
        lastTime = Time.time;
    }

    void PublishIMUData(Vector3 linearAcc, Vector3 angularVel)
    {
        // Publish to ROS or Unity event system
    }
}
```

## Sensor Fusion

### Combining Multiple Sensors
Sensor fusion combines data from multiple sensors to improve perception accuracy:

#### Example: Camera-LiDAR Fusion
```python
import numpy as np
import cv2

class SensorFusion:
    def __init__(self):
        # Camera intrinsic parameters
        self.camera_matrix = np.array([[fx, 0, cx],
                                      [0, fy, cy],
                                      [0, 0, 1]])

        # LiDAR to camera extrinsic parameters (rotation and translation)
        self.lidar_to_camera = np.eye(4)  # 4x4 transformation matrix

    def project_lidar_to_camera(self, lidar_points, camera_image):
        """Project LiDAR points onto camera image"""
        # Convert LiDAR points to homogeneous coordinates
        points_homo = np.hstack([lidar_points[:, :3], np.ones((len(lidar_points), 1))])

        # Transform points to camera frame
        points_cam = (self.lidar_to_camera @ points_homo.T).T

        # Project to image coordinates
        points_2d = (self.camera_matrix @ points_cam[:, :3].T).T
        points_2d = points_2d[:, :2] / points_2d[:, 2:3]  # Normalize by z

        # Filter points within image bounds
        valid_points = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < camera_image.shape[1]) & \
                       (points_2d[:, 1] >= 0) & (points_2d[:, 1] < camera_image.shape[0]) & \
                       (points_cam[:, 2] > 0)  # Points in front of camera

        return points_2d[valid_points], points_cam[valid_points, 2]  # 2D coordinates and depths
```

## Noise Modeling

### Realistic Sensor Noise
Real sensors have various types of noise that must be simulated:

1. **Gaussian Noise**: Random variations in measurements
2. **Bias**: Systematic offset in measurements
3. **Drift**: Slow changes in sensor characteristics over time
4. **Quantization**: Discrete representation of continuous signals

### Noise Implementation
```csharp
public class SensorNoise
{
    public static float AddGaussianNoise(float value, float stdDev, float mean = 0.0f)
    {
        // Box-Muller transform
        float u1 = Random.value;
        float u2 = Random.value;
        float gaussian = Mathf.Sqrt(-2.0f * Mathf.Log(u1)) * Mathf.Cos(2.0f * Mathf.PI * u2);
        return value + gaussian * stdDev + mean;
    }

    public static float AddBias(float value, float bias)
    {
        return value + bias;
    }

    public static float AddQuantization(float value, float stepSize)
    {
        return Mathf.Round(value / stepSize) * stepSize;
    }
}
```

## Calibration

### Sensor Calibration
Calibration ensures accurate sensor measurements:

#### Camera Calibration
```python
import cv2
import numpy as np

def calibrate_camera(images, pattern_size=(9, 6)):
    """Calibrate camera using checkerboard pattern"""
    obj_points = []  # 3D points in real world space
    img_points = []  # 2D points in image plane

    # Prepare object points
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            obj_points.append(objp)
            img_points.append(corners)

    # Perform calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None
    )

    return camera_matrix, dist_coeffs
```

## Best Practices

1. **Realistic Noise**: Include appropriate noise models for each sensor type
2. **Computational Efficiency**: Balance simulation accuracy with performance
3. **Validation**: Compare simulation results with real sensor data
4. **Parameter Tuning**: Adjust simulation parameters to match real sensor characteristics
5. **Cross-Sensor Validation**: Ensure consistency between different simulated sensors
6. **Environmental Factors**: Consider lighting, weather, and other environmental effects

## Practical Exercise

Create a simulation with:
1. LiDAR sensor detecting objects in the environment
2. RGB camera capturing visual information
3. IMU providing inertial measurements
4. Sensor fusion combining data from multiple sensors
5. Noise modeling for realistic sensor behavior

## Summary

- Sensor simulation is crucial for realistic robot perception testing
- Different sensor types require specific simulation approaches
- Noise modeling adds realism to sensor data
- Sensor fusion combines multiple sensor inputs
- Calibration ensures accurate sensor measurements
- Validation against real sensors ensures simulation quality