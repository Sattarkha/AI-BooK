---
sidebar_position: 1
---

# ROS 2 Nodes, Topics, and Services

## Overview

In ROS 2, communication between different parts of a robot system happens through three primary mechanisms: nodes, topics, and services. Understanding these concepts is fundamental to building distributed robotic applications.

## Nodes

A **node** is a process that performs computation. Nodes are the fundamental building blocks of a ROS 2 system. Multiple nodes work together to form a complete robotic application.

### Key Characteristics of Nodes:
- Each node runs a specific task or function
- Nodes communicate with each other through topics and services
- Nodes can be written in different programming languages
- Nodes are managed by the ROS 2 runtime system

### Creating a Node in Python:
```python
import rclpy
from rclpy.node import Node

class MinimalNode(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.get_logger().info('Minimal node created')

def main(args=None):
    rclpy.init(args=args)
    minimal_node = MinimalNode()

    try:
        rclpy.spin(minimal_node)
    except KeyboardInterrupt:
        pass
    finally:
        minimal_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Topics

**Topics** enable asynchronous communication between nodes through a publish-subscribe model. Publishers send messages to topics, and subscribers receive messages from topics.

### Key Characteristics of Topics:
- Asynchronous communication (publisher and subscriber don't need to be active simultaneously)
- Multiple publishers and subscribers can exist for the same topic
- Data flows from publishers to subscribers
- Topics are identified by names (e.g., `/cmd_vel`, `/sensor_data`)

### Publisher Example:
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1
```

### Subscriber Example:
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
```

## Services

**Services** enable synchronous communication between nodes through a request-response model. A client sends a request to a service and waits for a response.

### Key Characteristics of Services:
- Synchronous communication (client waits for response)
- Request-response pattern
- Only one service server can exist for each service name
- Services are identified by names (e.g., `/add_two_ints`, `/get_map`)

### Service Server Example:
```python
from add_two_ints_srv.srv import AddTwoInts
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
```

### Service Client Example:
```python
from add_two_ints_srv.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalClient(Node):
    def __init__(self):
        super().__init__('minimal_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')

    def send_request(self, a, b):
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        request = AddTwoInts.Request()
        request.a = a
        request.b = b
        self.future = self.cli.call_async(request)
```

## Actions

**Actions** are a third communication pattern that combines features of topics and services. They're used for long-running tasks with feedback.

### Key Characteristics of Actions:
- Used for tasks that take a long time to complete
- Provide feedback during execution
- Can be canceled
- Follow a goal-result-feedback pattern

## Practical Exercise

Create a simple ROS 2 package with a publisher and subscriber:

1. Create a new package: `ros2 pkg create --build-type ament_python my_robot_basics`
2. Implement a publisher that sends velocity commands
3. Implement a subscriber that receives sensor data
4. Test the communication between nodes

## Summary

- **Nodes** are the basic computational units in ROS 2
- **Topics** enable asynchronous communication through publish-subscribe
- **Services** enable synchronous communication through request-response
- **Actions** handle long-running tasks with feedback

Understanding these concepts is crucial for developing distributed robotic systems with ROS 2.