import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/dev/Desktop/HACK-60-/ros/install/gesture_control'
