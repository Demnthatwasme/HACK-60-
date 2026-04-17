import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
import os
import time
import math
from collections import deque, Counter
import torch.nn.functional as F

# ROS 2 Imports
import rclpy
from std_msgs.msg import String
from op3_walking_module_msgs.msg import WalkingParam
from op3_walking_module_msgs.srv import GetWalkingParam

# 1. AI Brain Architecture (Matches your trained model)
class GestureNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(GestureNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.network(x)

# 2. Robot Controller
class AIRobotController:
    def __init__(self, node):
        self.node = node
        self.cmd_pub   = self.node.create_publisher(String, '/robotis/walking/command', 10)
        self.param_pub = self.node.create_publisher(WalkingParam, '/robotis/walking/set_params', 10)
        self.ctrl_pub  = self.node.create_publisher(String, '/robotis/enable_ctrl_module', 10)
        self.param_srv = self.node.create_client(GetWalkingParam, '/robotis/walking/get_params')
        
        self.current_active_gesture = "STOP" 
        self.is_moving = False

        print("🤖 System Initializing... Enabling Walking Module.")
        self.ctrl_pub.publish(String(data='walking_module'))

    def execute_gesture(self, new_gesture):
        # Ignore UNKNOWN and duplicate commands (sticky logic)
        if new_gesture == "UNKNOWN" or new_gesture == self.current_active_gesture:
            return

        print(f"🔄 Transitioning: {self.current_active_gesture} -> {new_gesture}")

        # Stabilization: stop robot if currently moving before switching commands
        if self.is_moving:
            print("🛑 Stopping to settle posture...")
            self.cmd_pub.publish(String(data='stop'))
            time.sleep(2.0)
            self.is_moving = False

        if new_gesture == "STOP":
            self.current_active_gesture = "STOP"
            return

        print(f"🚀 Starting Motion: {new_gesture}")
        self.cmd_pub.publish(String(data='start'))
        self.is_moving = True
        
        if not self.param_srv.wait_for_service(timeout_sec=2.0):
            print("⚠️ Walking service timed out!")
            return

        req = GetWalkingParam.Request(get_param=True)
        future = self.param_srv.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        
        if future.result() is not None:
            params = future.result().parameters

            # FIX: class names must match classes.npy exactly: TURN_LEFT / TURN_RIGHT
            if new_gesture == "FORWARD":
                params.x_move_amplitude    = 0.025
                params.angle_move_amplitude = 0.0
            elif new_gesture == "TURN_LEFT":               # was "LEFT" — FIXED
                params.x_move_amplitude    = 0.0
                params.angle_move_amplitude = 0.2          # matches confirmed robot_control.py value
            elif new_gesture == "TURN_RIGHT":              # was "RIGHT" — FIXED
                params.x_move_amplitude    = 0.0
                params.angle_move_amplitude = -0.2         # matches confirmed robot_control.py value
                
            self.param_pub.publish(params)
            self.current_active_gesture = new_gesture
            print(f"✅ Physics Updated: {new_gesture}")

# 3. Setup AI and MediaPipe Tasks
current_dir = os.path.dirname(__file__)
classes = np.load(os.path.join(current_dir, 'classes.npy'), allow_pickle=True)
model = GestureNet(input_size=63, num_classes=len(classes))
model.load_state_dict(torch.load(os.path.join(current_dir, 'gesture_dl_model.pth')))
model.eval()

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=os.path.join(current_dir, 'hand_landmarker.task')),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

def main():
    rclpy.init()
    ros_node = rclpy.create_node('ai_gesture_controller')
    robot = AIRobotController(ros_node)
    
    cap = cv2.VideoCapture(0)
    prediction_buffer = deque(maxlen=7)

    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = landmarker.detect(mp_image)

            stable_command = "UNKNOWN"
            if detection_result.hand_landmarks:
                landmarks = detection_result.hand_landmarks[0]
                wrist = landmarks[0]
                mid_k = landmarks[9]
                bone_len = math.sqrt((mid_k.x-wrist.x)**2 + (mid_k.y-wrist.y)**2)
                bone_len = max(bone_len, 0.05)

                coords = []
                for lm in landmarks:
                    coords.extend([(lm.x-wrist.x)/bone_len, (lm.y-wrist.y)/bone_len, (lm.z-wrist.z)/bone_len])

                with torch.no_grad():
                    input_tensor = torch.FloatTensor([coords])
                    outputs = model(input_tensor)
                    prob = F.softmax(outputs, dim=1)
                    conf, pred_idx = torch.max(prob, 1)
                    raw_pred = classes[pred_idx.item()] if conf.item() > 0.85 else "UNKNOWN"

                prediction_buffer.append(raw_pred)
                stable_command = Counter(prediction_buffer).most_common(1)[0][0]

            # Actuate the robot
            robot.execute_gesture(stable_command)

            # Visual Feedback
            cv2.putText(frame, f"CMD: {stable_command}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Explainable AI Robot Controller', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    ros_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()