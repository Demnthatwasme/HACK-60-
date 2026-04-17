import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist


class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')

        self.subscription = self.create_subscription(
            String,
            'gesture_cmd',
            self.listener_callback,
            10
        )

        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)

        self.current_gesture = "STOP"

        # Publish continuously at 10 Hz
        self.timer = self.create_timer(0.1, self.publish_motion)

    def listener_callback(self, msg):
        self.current_gesture = msg.data
        self.get_logger().info(f"Gesture: {self.current_gesture}")

    def publish_motion(self):
        twist = Twist()

        if self.current_gesture == "FORWARD":
            twist.linear.x = 0.5

        elif self.current_gesture == "STOP":
            twist.linear.x = 0.0

        elif self.current_gesture == "TURN_LEFT":
            twist.angular.z = 0.5

        elif self.current_gesture == "TURN_RIGHT":
            twist.angular.z = -0.5

        self.publisher_.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()