Terminal 1:
source /opt/ros/humble/setup.bash
cd ~/ros
source install/setup.bash
ros2 launch turtlebot3_gazebo empty_world.launch.py
Terminal 2:
source /opt/ros/humble/setup.bash
cd ~/ros
source install/setup.bash
<!-- ros2 run gesture_control robot_ctrl -->
/usr/bin/python3 /home/dev/Desktop/HACK-60-/ros/src/gesture_control/gesture_control/robot_controller.py

Terminal 3:
source /opt/ros/humble/setup.bash
cd ~/ros
source install/setup.bash
<!-- ros2 run gesture_control gesture_pub -->
/usr/bin/python3 /home/dev/Desktop/HACK-60-/ros/src/gesture_control/gesture_control/gesture_publisher.py

Terminal 4 (debug):
ros2 topic echo /gesture_cmd

Copy these into:
ros/src/gesture_control/gesture_control/
Files:
gesture_dl_model.pth
classes.npy
hand_landmarker.task

sudo apt install python3-lxml

QUICK DEBUG (if still not visible)

Run:

ros2 topic list

You should see:

/spawn_entity
/cmd_vel
/odom
/scan

CLEAN RESTART (important)

Do this properly:

killall gzserver gzclient

source /opt/ros/humble/setup.bash

export TURTLEBOT3_MODEL=burger

ros2 launch turtlebot3_gazebo empty_world.launch.py


# Deactivate conda first to use /usr/bin/python3 
conda deactivate