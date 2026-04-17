import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
import os
from collections import deque, Counter
import torch.nn.functional as F
import statistics
import math

# ── ADD 1: ROS 2 imports & publisher setup ───────────────────────────────
import rclpy
from std_msgs.msg import String

rclpy.init()
ros_node  = rclpy.create_node('ai_gesture_publisher')
publisher = ros_node.create_publisher(String, '/gesture_command', 10)
last_published = None   # avoid flooding the topic with repeated identical commands
# ────────────────────────────────────────────────────────────────────────

print("1. Loading AI Brain...")
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
    num_hands=2
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

print("2. Starting Live Inference...")
prediction_buffer = deque(maxlen=7)

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = landmarker.detect(mp_image)

        if detection_result.hand_landmarks:

            target_hand_landmarks = None
            for i, handedness_list in enumerate(detection_result.handedness):
                if handedness_list[0].category_name == "Left":
                    target_hand_landmarks = detection_result.hand_landmarks[i]
                    break
            if target_hand_landmarks is None:
                target_hand_landmarks = detection_result.hand_landmarks[0]

            wrist         = target_hand_landmarks[0]
            middle_knuckle = target_hand_landmarks[9]
            bone_length = math.sqrt(
                (middle_knuckle.x - wrist.x)**2 +
                (middle_knuckle.y - wrist.y)**2 +
                (middle_knuckle.z - wrist.z)**2
            )
            bone_length = max(bone_length, 0.05)

            coords = []
            for lm in target_hand_landmarks:
                coords.extend([
                    (lm.x - wrist.x) / bone_length,
                    (lm.y - wrist.y) / bone_length,
                    (lm.z - wrist.z) / bone_length,
                ])

            input_tensor = torch.FloatTensor([coords])
            input_tensor.requires_grad = True

            outputs      = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence    = torch.max(probabilities).item() * 100
            predicted_idx = torch.argmax(outputs).item()
            prediction_label = classes[predicted_idx]

            raw_prediction = prediction_label if confidence >= 85.0 else "UNKNOWN"
            prediction_buffer.append(raw_prediction)

            counts = Counter(prediction_buffer)
            most_common_pred, count = counts.most_common(1)[0]
            stable_command = most_common_pred if count >= 4 else "UNKNOWN"

            # ── ADD 2: Publish stable_command to the robot ───────────────
            #global last_published
            if stable_command != "UNKNOWN" and stable_command != last_published:
                msg = String()
                msg.data = stable_command
                publisher.publish(msg)
                last_published = stable_command
                print(f"  ▶ Published: {stable_command}")
            # ────────────────────────────────────────────────────────────

            if stable_command == "UNKNOWN":
                display_text = "Command: --- (Waiting)"
                text_color   = (150, 150, 150)
            else:
                display_text = f"Command: {stable_command}"
                text_color   = (0, 255, 0)

            model.zero_grad()
            outputs[0, predicted_idx].backward()
            gradients        = input_tensor.grad.abs().numpy()[0]
            joint_importances = [sum(gradients[i*3:(i*3)+3]) for i in range(21)]
            max_importance    = max(joint_importances) if max(joint_importances) > 0 else 1

            for connection in HAND_CONNECTIONS:
                s, e = connection
                sp = (int(target_hand_landmarks[s].x * frame.shape[1]),
                      int(target_hand_landmarks[s].y * frame.shape[0]))
                ep = (int(target_hand_landmarks[e].x * frame.shape[1]),
                      int(target_hand_landmarks[e].y * frame.shape[0]))
                cv2.line(frame, sp, ep, (200, 200, 200), 2)

            for i, landmark in enumerate(target_hand_landmarks):
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                intensity = joint_importances[i] / max_importance
                r  = int(255 * intensity)
                b  = int(255 * (1 - intensity))
                radius = int(5 + (10 * intensity))
                cv2.circle(frame, (x, y), radius, (b, 0, r), -1)
                cv2.circle(frame, (x, y), radius, (255, 255, 255), 1)

            cv2.putText(frame, display_text, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3, cv2.LINE_AA)
            cv2.putText(frame, f'AI Confidence: {confidence:.1f}%', (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 200), 2, cv2.LINE_AA)

        cv2.imshow('Explainable AI - Robot Controller', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ── ADD 3: ROS 2 cleanup ─────────────────────────────────────────────────
ros_node.destroy_node()
rclpy.shutdown()
# ────────────────────────────────────────────────────────────────────────