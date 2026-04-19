import cv2
import mediapipe as mp
import numpy as np
import threading
import asyncio
import os
import time
from aiohttp import web
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pose_landmarker_heavy.task')
current_data = {"landmarks": [], "posture": "Unknown"}

# --- HARDCODED SKELETON MAP ---
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),       # Right Eye
    (0, 4), (4, 5), (5, 6), (6, 8),       # Left Eye
    (9, 10),                              # Mouth
    (11, 12),                             # Shoulders
    (11, 13), (13, 15),                   # Left Arm
    (15, 17), (15, 19), (15, 21), (17, 19),# Left Hand
    (12, 14), (14, 16),                   # Right Arm
    (16, 18), (16, 20), (16, 22), (18, 20),# Right Hand
    (11, 23), (12, 24), (23, 24),         # Torso
    (23, 25), (25, 27),                   # Left Leg
    (27, 29), (29, 31), (27, 31),         # Left Foot
    (24, 26), (26, 28),                   # Right Leg
    (28, 30), (30, 32), (28, 32)          # Right Foot
]

def calculate_angle(a, b, c):
    """Calculates the angle at joint 'b' given points a, b, c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def dist(p1, p2):
    """Calculates 3D Euclidean distance between two points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def classify_pose(landmarks):
    """Classifies posture using robust 3D distance math."""
    nose = landmarks[0] 
    l_sh, r_sh = landmarks[11], landmarks[12]
    l_wr, r_wr = landmarks[15], landmarks[16]
    l_hip, r_hip = landmarks[23], landmarks[24]
    l_kny, r_kny = landmarks[25], landmarks[26]
    l_ank, r_ank = landmarks[27], landmarks[28]

    # 1. Hands Up/Down 
    l_hand_up = l_wr[1] < l_sh[1]
    r_hand_up = r_wr[1] < r_sh[1]
    
    # 2. Clap Detection
    is_clapping = dist(l_wr, r_wr) < 0.15

    # 3. Angle and Height Calculations for Lower Body
    l_knee_angle = calculate_angle(l_hip, l_kny, l_ank)
    r_knee_angle = calculate_angle(r_hip, r_kny, r_ank)
    
    # NEW: Calculate Hip Angles (Torso to Thigh)
    l_hip_angle = calculate_angle(l_sh, l_hip, l_kny)
    r_hip_angle = calculate_angle(r_sh, r_hip, r_kny)

    hip_height = (l_hip[1] + r_hip[1]) / 2
    ank_height = (l_ank[1] + r_ank[1]) / 2
    height_diff = ank_height - hip_height 

    # 4. T-Pose 
    l_arm_out = dist(l_wr, l_sh) > 0.4
    r_arm_out = dist(r_wr, r_sh) > 0.4
    arms_level = abs(l_wr[1] - l_sh[1]) < 0.2 and abs(r_wr[1] - r_sh[1]) < 0.2
    is_t_pose = l_arm_out and r_arm_out and arms_level

    # 5. Dab Pose 
    right_dab = (dist(l_wr, l_sh) > 0.3) and l_hand_up and (dist(r_wr, nose) < 0.2)
    left_dab = (dist(r_wr, r_sh) > 0.3) and r_hand_up and (dist(l_wr, nose) < 0.2)
    is_dabbing = right_dab or left_dab

    # 6. Karate Pose 
    left_leg_karate = l_ank[1] < (r_kny[1] + 0.15)
    right_leg_karate = r_ank[1] < (l_kny[1] + 0.15)
    is_karate = left_leg_karate or right_leg_karate

    # --- NEW: SQUAT VS SIT LOGIC ---
    # Squat: Deep knee bend AND hips are vertically very close to the ankles
    is_squat = (l_knee_angle < 100 and r_knee_angle < 100) and (height_diff < 0.4)

    # Sit: Hips bent, Knees bent, AND Knees push forward in Z-space (closer to camera)
    knees_forward = (l_kny[2] < l_hip[2] - 0.1) or (r_kny[2] < r_hip[2] - 0.1)
    hips_bent = (l_hip_angle < 130) or (r_hip_angle < 130)
    knees_bent = (l_knee_angle < 130) or (r_knee_angle < 130)
    is_sitting = hips_bent and knees_bent and knees_forward and not is_squat

    # --- DECISION TREE ---
    if is_clapping: return "Clap"
    if is_dabbing: return "Dab Pose"
    if is_t_pose: return "T-Pose"
    if is_karate: return "Karate Pose"
    
    if l_hand_up and r_hand_up: return "Both Hands Up"
    if l_hand_up: return "Left Hand Up"
    if r_hand_up: return "Right Hand Up"
    
    # Updated Lower Body Prioritization
    if is_squat: return "Squats/Crouch"
    if is_sitting: return "Sitting" 
    
    if l_sh[2] < -0.3: return "Leaning Forward"
    if l_sh[2] > 0.2: return "Leaning Backward"
    
    return "Standing"

def vision_loop():
    global current_data
    if not os.path.exists(MODEL_PATH): 
        print(f"Error: {MODEL_PATH} not found.")
        return

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    
    # Updated to read from the RAMDisk broadcasted by the main pipeline!
    # This prevents the two scripts from fighting over the single DroidCam connection.
    while True:
        frame = cv2.imread('/dev/shm/droidcam_frame.jpg')
        if frame is None:
            print("VIBE: Waiting for main pipeline to broadcast frame...")
            time.sleep(0.5)
            continue

        # 1. INFERENCE ON RAW FRAME (Ensures Left/Right data is perfectly accurate)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        result = detector.detect_for_video(mp_image, timestamp_ms)

        # 2. Extract dimensions
        h, w, _ = frame.shape

        if result.pose_world_landmarks:
            world_landmarks = result.pose_world_landmarks[0]
            pts = [[lm.x, lm.y, lm.z] for lm in world_landmarks]
            
            label = classify_pose(pts)
            current_data["landmarks"] = pts
            current_data["posture"] = label
            
            cv2.putText(frame, f"Pose: {label}", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

            if result.pose_landmarks:
                norm_landmarks = result.pose_landmarks[0]
                
                # Draw the Skeleton Lines
                for connection in POSE_CONNECTIONS:
                    start_idx, end_idx = connection
                    sx = int(norm_landmarks[start_idx].x * w)
                    sy = int(norm_landmarks[start_idx].y * h)
                    ex = int(norm_landmarks[end_idx].x * w)
                    ey = int(norm_landmarks[end_idx].y * h)
                    cv2.line(frame, (sx, sy), (ex, ey), (255, 255, 255), 2) 

                # Draw the Joint Dots
                for lm in norm_landmarks:
                    cx = int(lm.x * w)
                    cy = int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        cv2.imshow('Vibe Classifier', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): 
            break
        elif key == ord('r'):
            # Send reset signal to the main pipeline via RAMDisk flag
            open('/dev/shm/reset_robot', 'w').close()
            print("VIBE: Sent reset signal to main pipeline!")

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