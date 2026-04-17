#!/usr/bin/env python3
"""
keyboard_publisher.py
Publishes gesture commands to /gesture_command via arrow keys.
Arrow Left  → 'left'   (head turns left)
Arrow Right → 'right'  (head turns right)
Space/Up    → 'center' (head returns to center)
Q           → quit
"""

import sys
import tty
import termios
import rclpy
from std_msgs.msg import String

# --- Key byte sequences (Linux terminal) ---
KEY_LEFT   = '\x1b[D'
KEY_RIGHT  = '\x1b[C'
KEY_UP     = '\x1b[A'
KEY_SPACE  = ' '
KEY_QUIT   = 'q'

def get_key(settings):
    """Reads a single keypress (blocking), returns string."""
    tty.setraw(sys.stdin.fileno())
    key = sys.stdin.read(1)
    # Arrow keys send 3-byte escape sequences
    if key == '\x1b':
        key += sys.stdin.read(2)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def main():
    rclpy.init()
    node = rclpy.create_node('keyboard_gesture_publisher')
    publisher = node.create_publisher(String, '/gesture_command', 10)

    print("🎮 Keyboard Publisher Ready")
    print("  ← / →  : Turn head left / right")
    print("  ↑ / SPC : Center head")
    print("  Q       : Quit\n")

    original_settings = termios.tcgetattr(sys.stdin)

    try:
        while rclpy.ok():
            key = get_key(original_settings)

            if key == KEY_LEFT:
                command = 'left'
            elif key == KEY_RIGHT:
                command = 'right'
            elif key in (KEY_UP, KEY_SPACE):
                command = 'center'
            elif key.lower() == KEY_QUIT:
                print("Shutting down.")
                break
            else:
                continue  # Ignore unmapped keys

            msg = String()
            msg.data = command
            publisher.publish(msg)
            print(f"  ▶ Published: '{command}'")

    finally:
        # Always restore terminal on exit
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original_settings)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()