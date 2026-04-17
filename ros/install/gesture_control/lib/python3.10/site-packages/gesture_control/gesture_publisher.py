import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F


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


class GesturePublisher(Node):
    def __init__(self):
        super().__init__('gesture_publisher')

        self.publisher_ = self.create_publisher(String, 'gesture_cmd', 10)

        # Load model
        current_dir = os.path.dirname(__file__)
        self.classes = np.load(os.path.join(current_dir, 'classes.npy'), allow_pickle=True)

        self.model = GestureNet(63, len(self.classes))
        self.model.load_state_dict(torch.load(os.path.join(current_dir, 'gesture_dl_model.pth')))
        self.model.eval()

        # MediaPipe setup
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self.options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=os.path.join(current_dir, 'hand_landmarker.task')),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=1
        )

        self.cap = cv2.VideoCapture(0)

        self.timer = self.create_timer(0.05, self.process_frame)  # ~20 FPS

        self.landmarker = HandLandmarker.create_from_options(self.options)

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.landmarker.detect(mp_image)

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]

            wrist = lm[0]
            coords = []

            for p in lm:
                coords.extend([
                    p.x - wrist.x,
                    p.y - wrist.y,
                    p.z - wrist.z
                ])

            input_tensor = torch.FloatTensor([coords])
            outputs = self.model(input_tensor)

            probs = F.softmax(outputs, dim=1)
            confidence = torch.max(probs).item() * 100
            pred = torch.argmax(outputs).item()
            label = self.classes[pred]

            if confidence > 85 and label != "UNKNOWN":
                msg = String()
                msg.data = label
                self.publisher_.publish(msg)

                self.get_logger().info(f'Published: {label}')

        cv2.imshow("Gesture", frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = GesturePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()