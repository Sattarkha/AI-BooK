---
sidebar_position: 2
---

# Python Integration with ROS 2 (rclpy)

## Overview

Python is one of the most popular languages for robotics development due to its simplicity and the extensive ecosystem of scientific computing libraries. ROS 2 provides the `rclpy` package for Python integration, which allows you to create nodes, publish and subscribe to topics, and use services and actions.

## Installing rclpy

To use rclpy, you need to have ROS 2 installed. The rclpy package is typically included in the ROS 2 installation:

```bash
# Make sure ROS 2 is sourced
source /opt/ros/humble/setup.bash  # or your ROS 2 distribution

# Check if rclpy is available
python3 -c "import rclpy; print('rclpy available')"
```

## Basic Node Structure

Every ROS 2 Python node follows a similar structure:

```python
import rclpy
from rclpy.node import Node

class MyNode(Node):
    def __init__(self):
        super().__init__('node_name')
        # Node initialization code here
        self.get_logger().info('Node has been initialized')

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Creating Publishers

Publishers send messages to topics:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Talker(Node):
    def __init__(self):
        super().__init__('talker')
        self.publisher = self.create_publisher(String, 'chatter', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    talker = Talker()

    try:
        rclpy.spin(talker)
    except KeyboardInterrupt:
        pass
    finally:
        talker.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Creating Subscribers

Subscribers receive messages from topics:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Listener(Node):
    def __init__(self):
        super().__init__('listener')
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    listener = Listener()

    try:
        rclpy.spin(listener)
    except KeyboardInterrupt:
        pass
    finally:
        listener.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Creating Services

Services allow for request-response communication:

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

    try:
        rclpy.spin(minimal_service)
    except KeyboardInterrupt:
        pass
    finally:
        minimal_service.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Service Client

Creating a client that calls a service:

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalClient(Node):
    def __init__(self):
        super().__init__('minimal_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

    def send_request(self, a, b):
        request = AddTwoInts.Request()
        request.a = a
        request.b = b
        self.future = self.cli.call_async(request)

def main(args=None):
    rclpy.init(args=args)
    minimal_client = MinimalClient()

    # Send a request
    minimal_client.send_request(1, 2)

    while rclpy.ok():
        rclpy.spin_once(minimal_client)
        if minimal_client.future.done():
            try:
                response = minimal_client.future.result()
            except Exception as e:
                minimal_client.get_logger().info(f'Service call failed: {e}')
            else:
                minimal_client.get_logger().info(f'Result: {response.sum}')
            break

    minimal_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Working with Parameters

ROS 2 nodes can have parameters that can be configured at runtime:

```python
import rclpy
from rclpy.node import Node

class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('my_parameter', 'default_value')
        self.declare_parameter('count_threshold', 5)

        # Get parameter values
        self.my_param = self.get_parameter('my_parameter').value
        self.threshold = self.get_parameter('count_threshold').value

        self.get_logger().info(f'My parameter: {self.my_param}')
        self.get_logger().info(f'Threshold: {self.threshold}')

def main(args=None):
    rclpy.init(args=args)
    node = ParameterNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Timer-based Execution

Timers allow you to execute code at regular intervals:

```python
import rclpy
from rclpy.node import Node

class TimerNode(Node):
    def __init__(self):
        super().__init__('timer_node')
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.counter = 0

    def timer_callback(self):
        self.get_logger().info(f'Timer callback: {self.counter}')
        self.counter += 1

def main(args=None):
    rclpy.init(args=args)
    node = TimerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Creating Custom Messages

To create custom message types:

1. Create a `msg` directory in your package
2. Define your message in a `.msg` file:

```
# In MyMessage.msg
string name
int32 id
float64 value
bool active
```

3. Add the message to your CMakeLists.txt and package.xml
4. Use the message in your Python code:

```python
from my_package.msg import MyMessage  # Replace with your package name

# In your node
msg = MyMessage()
msg.name = "test"
msg.id = 123
msg.value = 3.14
msg.active = True
```

## Best Practices

1. **Error Handling**: Always include proper error handling in your nodes
2. **Logging**: Use the node's logger for debugging and monitoring
3. **Resource Management**: Properly destroy nodes and clean up resources
4. **Parameter Validation**: Validate parameters before using them
5. **Threading**: Be aware of threading issues in ROS 2 Python nodes

## Practical Exercise

Create a Python node that:
1. Subscribes to a topic with sensor data
2. Publishes processed data to another topic
3. Provides a service to reset internal state
4. Uses parameters to configure behavior

## Summary

- `rclpy` is the Python client library for ROS 2
- Follow the standard node structure with proper initialization and cleanup
- Use publishers for asynchronous communication and services for synchronous communication
- Parameters allow runtime configuration of nodes
- Timers enable periodic execution of code
- Custom messages can be created for domain-specific data