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

    def listener_callback(self, msg):
        gesture = msg.data
        twist = Twist()

        if gesture == "FORWARD":
            twist.linear.x = 0.5

        elif gesture == "STOP":
            twist.linear.x = 0.0

        elif gesture == "TURN_LEFT":
            twist.angular.z = 0.5

        elif gesture == "TURN_RIGHT":
            twist.angular.z = -0.5

        elif gesture == "GRAB":
            self.get_logger().info("GRAB action triggered")

        self.publisher_.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()