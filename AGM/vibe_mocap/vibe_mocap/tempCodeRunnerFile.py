import cv2
import mediapipe as mp
import numpy as np
import threading
import asyncio
import os
from aiohttp import web
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
MODEL_PATH = 'pose_landmarker_heavy.task'
current_data = {"landmarks": [], "posture": "Unknown"}

def calculate_angle(a, b, c):
    """Calculates the angle at joint 'b' given points a, b, c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def classify_pose(landmarks):
    """
    Classifies posture based on MediaPipe World Landmarks (meters).
    Index Map: 11,12=Shoulders | 13,14=Elbows | 15,16=Wrists | 23,24=Hips | 27,28=Ankles
    """
    l_sh, r_sh = landmarks[11], landmarks[12]
    l_wr, r_wr = landmarks[15], landmarks[16]
    l_hip, r_hip = landmarks[23], landmarks[24]
    l_ank, r_ank = landmarks[27], landmarks[28]
    l_kny, r_kny = landmarks[25], landmarks[26]

    # 1. Hands Up/Down
    hands_up = l_wr[1] < l_sh[1] and r_wr[1] < r_sh[1]
    
    # 2. Clap Detection (Wrists close together)
    dist_wrists = np.linalg.norm(np.array(l_wr) - np.array(r_wr))
    is_clapping = dist_wrists < 0.15

    # 3. Squat/Crouch Detection
    # Calculate Knee Angles
    l_knee_angle = calculate_angle(l_hip, l_kny, l_ank)
    r_knee_angle = calculate_angle(r_hip, r_kny, r_ank)
    
    # Average Hip Height relative to Ankles
    hip_height = (l_hip[1] + r_hip[1]) / 2
    ank_height = (l_ank[1] + r_ank[1]) / 2
    height_diff = ank_height - hip_height # MediaPipe Y is down

    if is_clapping: return "Clap"
    if hands_up: return "Hands Up"
    if l_knee_angle < 100 and r_knee_angle < 100: return "Squats/Crouch"
    if height_diff < 0.5: return "Sitting" # Threshold for sitting
    if l_sh[2] < -0.3: return "Leaning Forward"
    if l_sh[2] > 0.3: return "Leaning Backward"
    
    return "Standing"

def vision_loop():
    global current_data
    if not os.path.exists(MODEL_PATH): return

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success: continue

        frame = cv2.flip(frame, 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        result = detector.detect_for_video(mp_image, timestamp_ms)

        if result.pose_world_landmarks:
            world_landmarks = result.pose_world_landmarks[0]
            # Convert landmarks to list of [x, y, z]
            pts = [[lm.x, lm.y, lm.z] for lm in world_landmarks]
            
            # RUN CLASSIFIER
            label = classify_pose(pts)
            
            current_data["landmarks"] = pts
            current_data["posture"] = label
            
            # Display label on screen
            cv2.putText(frame, f"Pose: {label}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if result.pose_landmarks:
                for lm in result.pose_landmarks[0]:
                    cv2.circle(frame, (int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])), 2, (255, 0, 0), -1)

        cv2.imshow('Vibe Classifier', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

async def handle_get_pose(request):
    return web.json_response(current_data)

async def start_server():
    app = web.Application()
    app.router.add_get('/get_pose', handle_get_pose)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, 'localhost', 8080).start()

if __name__ == "__main__":
    threading.Thread(target=vision_loop, daemon=True).start()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(start_server())
        loop.run_forever()
    except KeyboardInterrupt:
        pass