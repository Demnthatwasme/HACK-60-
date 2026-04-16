import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F

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
    num_hands=2 # MediaPipe is allowed to see both hands
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         
    (0, 5), (5, 6), (6, 7), (7, 8),         
    (5, 9), (9, 10), (10, 11), (11, 12),    
    (9, 13), (13, 14), (14, 15), (15, 16),  
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)  
]

print("2. Starting Live Inference...")
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
            
            # --- PRIORITY LOGIC: Find the physical Right hand ---
            target_hand_landmarks = None
            
            for i, handedness_list in enumerate(detection_result.handedness):
                hand_label = handedness_list[0].category_name 
                
                # "Left" in a mirrored webcam is your physical right hand
                if hand_label == "Left":
                    target_hand_landmarks = detection_result.hand_landmarks[i]
                    break 
            
            if target_hand_landmarks is None:
                target_hand_landmarks = detection_result.hand_landmarks[0]
            # ----------------------------------------------------

            # --- NORMALIZED COORDINATE EXTRACTION ---
            wrist = target_hand_landmarks[0] # Anchor point
            coords = []
            
            for lm in target_hand_landmarks:
                # Subtract wrist position to make it relative!
                rel_x = lm.x - wrist.x
                rel_y = lm.y - wrist.y
                rel_z = lm.z - wrist.z
                coords.extend([rel_x, rel_y, rel_z])
            
            input_tensor = torch.FloatTensor([coords])
            input_tensor.requires_grad = True 
            # ----------------------------------------
            
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence = torch.max(probabilities).item() * 100
            predicted_idx = torch.argmax(outputs).item()
            prediction_label = classes[predicted_idx]
            
            # --- THE HYBRID GARBAGE FILTER ---
            if prediction_label == "UNKNOWN" or confidence < 85.0:
                display_text = "Command: --- (Waiting)"
                text_color = (150, 150, 150) # Gray
            else:
                display_text = f"Command: {prediction_label}"
                text_color = (0, 255, 0) # Green
            
            # --- EXPLAINABLE AI: SKELETAL HEATMAP ---
            model.zero_grad()
            outputs[0, predicted_idx].backward()
            gradients = input_tensor.grad.abs().numpy()[0]
            joint_importances = [sum(gradients[i*3:(i*3)+3]) for i in range(21)]
            max_importance = max(joint_importances) if max(joint_importances) > 0 else 1
            
            for connection in HAND_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                # Drawing still uses absolute coordinates so it matches the video frame
                start_point = (int(target_hand_landmarks[start_idx].x * frame.shape[1]), int(target_hand_landmarks[start_idx].y * frame.shape[0]))
                end_point = (int(target_hand_landmarks[end_idx].x * frame.shape[1]), int(target_hand_landmarks[end_idx].y * frame.shape[0]))
                cv2.line(frame, start_point, end_point, (200, 200, 200), 2)
            
            for i, landmark in enumerate(target_hand_landmarks):
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                intensity = joint_importances[i] / max_importance
                r = int(255 * intensity)
                b = int(255 * (1 - intensity))
                g = 0
                radius = int(5 + (10 * intensity))
                cv2.circle(frame, (x, y), radius, (b, g, r), -1)
                cv2.circle(frame, (x, y), radius, (255, 255, 255), 1) 
            
            # --- ON-SCREEN UI ---
            cv2.putText(frame, display_text, (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3, cv2.LINE_AA)
            cv2.putText(frame, f'AI Confidence: {confidence:.1f}%', (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 200), 2, cv2.LINE_AA)

        cv2.imshow('Explainable AI - Robot Controller', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()