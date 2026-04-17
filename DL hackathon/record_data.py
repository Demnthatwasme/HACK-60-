import cv2
import mediapipe as mp
import csv
import os

# --- SET YOUR CURRENT GESTURE HERE ---
GESTURE_LABEL = "STOP"  # Options: "STOP", "FORWARD", "TURN_LEFT", "TURN_RIGHT", "DAB" , "UNKNOWN"
# -------------------------------------

# 1. Setup the Modern Tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Initialize the landmarker 
# (Using os.path to ensure it always finds the model file regardless of terminal directory)
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

# 2. CSV File Preparation & Initial Counting
csv_file = 'gesture_dataset.csv'
file_exists = os.path.isfile(csv_file)
frame_count = 0

# If the file exists, count how many frames of THIS gesture we already have
if file_exists:
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0] == GESTURE_LABEL:
                frame_count += 1

with open(csv_file, mode='a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        header = ['label']
        for i in range(21):
            header.extend([f'x{i}', f'y{i}', f'z{i}'])
        writer.writerow(header)

    # 3. Create the Landmarker Instance
    with HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        print(f"Recording data for: {GESTURE_LABEL}")
        print("Press 's' to save a frame. Press 'q' to quit.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                break

            # Mirror the frame for a natural selfie-view
            frame = cv2.flip(frame, 1)
            
            # The Tasks API strictly requires an mp.Image object
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Perform detection
            detection_result = landmarker.detect(mp_image)

            # Extract data and draw visual feedback
            # Extract data and draw visual feedback
            if detection_result.hand_landmarks:
                # Just grab the very first hand detected on screen
                target_hand_landmarks = detection_result.hand_landmarks[0]

                # Draw green circles on the joints
                for landmark in target_hand_landmarks:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                # Listen for the save key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    row = [GESTURE_LABEL]
                    for landmark in target_hand_landmarks:
                        row.extend([landmark.x, landmark.y, landmark.z])
                    
                    writer.writerow(row)
                    frame_count += 1 
                    print(f"Total {GESTURE_LABEL} frames: {frame_count}")

            # --- NEW: ON-SCREEN DASHBOARD ---
            # Draw the current gesture label in blue
            cv2.putText(frame, f'Recording: {GESTURE_LABEL}', (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
            # Draw the live frame count in green
            cv2.putText(frame, f'Total Frames: {frame_count}', (10, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('Tasks API Dataset Creator', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()