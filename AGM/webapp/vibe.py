"""
vibe.py — MediaPipe posture classifier for G1 Mission Control.

All MediaPipe imports are wrapped in try/except.
If mediapipe is not installed, detect_posture() returns a mock annotated frame.

Source logic ported from: AGM/vibe_mocap/vibe_mocap/mocap.py
"""
import cv2
import numpy as np
import os
import time

# ── T2.1 — Optional MediaPipe import ─────────────────────────────────────────
MEDIAPIPE_OK = False
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_OK = True
    print("[vibe] MediaPipe: OK")
except ImportError:
    print("[vibe] MediaPipe not installed — mock mode active (VIBE will show placeholders)")

# ── T2.4 — Skeleton connectivity map (module-level) ──────────────────────────
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),           # Right Eye
    (0, 4), (4, 5), (5, 6), (6, 8),           # Left Eye
    (9, 10),                                   # Mouth
    (11, 12),                                  # Shoulders
    (11, 13), (13, 15),                        # Left Arm
    (15, 17), (15, 19), (15, 21), (17, 19),   # Left Hand
    (12, 14), (14, 16),                        # Right Arm
    (16, 18), (16, 20), (16, 22), (18, 20),   # Right Hand
    (11, 23), (12, 24), (23, 24),             # Torso
    (23, 25), (25, 27),                        # Left Leg
    (27, 29), (29, 31), (27, 31),             # Left Foot
    (24, 26), (26, 28),                        # Right Leg
    (28, 30), (30, 32), (28, 32),             # Right Foot
]


# ── T2.2 — Helper functions ───────────────────────────────────────────────────
def calculate_angle(a, b, c):
    """Calculates the angle (degrees) at joint 'b' given points a, b, c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


def dist(p1, p2):
    """3D Euclidean distance between two points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))


# ── T2.3 — classify_pose ─────────────────────────────────────────────────────
def classify_pose(landmarks):
    """
    Classifies posture from MediaPipe world landmarks.

    Input:  list of 33 [x, y, z] world-space landmark coordinates
    Output: str label (e.g. 'Standing', 'Squat', 'T-Pose', ...)

    Decision tree (priority order):
      Clap → Dab → T-Pose → Karate → Both Hands Up →
      Left Hand Up → Right Hand Up → Squat → Sit →
      Lean Forward → Lean Backward → Standing
    """
    nose  = landmarks[0]
    l_sh, r_sh   = landmarks[11], landmarks[12]
    l_wr, r_wr   = landmarks[15], landmarks[16]
    l_hip, r_hip = landmarks[23], landmarks[24]
    l_kny, r_kny = landmarks[25], landmarks[26]
    l_ank, r_ank = landmarks[27], landmarks[28]

    # Hands Up/Down
    l_hand_up = l_wr[1] < l_sh[1]
    r_hand_up = r_wr[1] < r_sh[1]

    # Clap
    is_clapping = dist(l_wr, r_wr) < 0.15

    # Knee / Hip angles
    l_knee_angle = calculate_angle(l_hip, l_kny, l_ank)
    r_knee_angle = calculate_angle(r_hip, r_kny, r_ank)
    l_hip_angle  = calculate_angle(l_sh,  l_hip, l_kny)
    r_hip_angle  = calculate_angle(r_sh,  r_hip, r_kny)

    hip_height  = (l_hip[1] + r_hip[1]) / 2
    ank_height  = (l_ank[1] + r_ank[1]) / 2
    height_diff = ank_height - hip_height

    # T-Pose
    l_arm_out  = dist(l_wr, l_sh) > 0.4
    r_arm_out  = dist(r_wr, r_sh) > 0.4
    arms_level = (abs(l_wr[1] - l_sh[1]) < 0.2) and (abs(r_wr[1] - r_sh[1]) < 0.2)
    is_t_pose  = l_arm_out and r_arm_out and arms_level

    # Dab
    right_dab  = (dist(l_wr, l_sh) > 0.3) and l_hand_up and (dist(r_wr, nose) < 0.2)
    left_dab   = (dist(r_wr, r_sh) > 0.3) and r_hand_up and (dist(l_wr, nose) < 0.2)
    is_dabbing = right_dab or left_dab

    # Karate
    left_leg_karate  = l_ank[1] < (r_kny[1] + 0.15)
    right_leg_karate = r_ank[1] < (l_kny[1] + 0.15)
    is_karate        = left_leg_karate or right_leg_karate

    # Squat
    is_squat = (l_knee_angle < 100 and r_knee_angle < 100) and (height_diff < 0.4)

    # Sit
    knees_forward = (l_kny[2] < l_hip[2] - 0.1) or (r_kny[2] < r_hip[2] - 0.1)
    hips_bent     = (l_hip_angle < 130) or (r_hip_angle < 130)
    knees_bent    = (l_knee_angle < 130) or (r_knee_angle < 130)
    is_sitting    = hips_bent and knees_bent and knees_forward and not is_squat

    # Decision tree
    if is_clapping: return "Clap"
    if is_dabbing:  return "Dab Pose"
    if is_t_pose:   return "T-Pose"
    if is_karate:   return "Karate Pose"
    if l_hand_up and r_hand_up: return "Both Hands Up"
    if l_hand_up:   return "Left Hand Up"
    if r_hand_up:   return "Right Hand Up"
    if is_squat:    return "Squats/Crouch"
    if is_sitting:  return "Sitting"
    if l_sh[2] < -0.3: return "Leaning Forward"
    if l_sh[2] > 0.2:  return "Leaning Backward"
    return "Standing"


# ── T2.5 — init_mediapipe_detector ───────────────────────────────────────────
def init_mediapipe_detector(model_dir=None):
    """
    Initialise MediaPipe PoseLandmarker in VIDEO mode.
    Returns detector or None (mock mode).

    Searches for 'pose_landmarker_heavy.task':
      1. model_dir argument
      2. ../vibe_mocap/vibe_mocap/ relative to this file
      3. current working directory
    """
    if not MEDIAPIPE_OK:
        print("[vibe] init_mediapipe_detector: MediaPipe unavailable — returning None")
        return None

    _candidates = []
    if model_dir:
        _candidates.append(os.path.join(model_dir, 'pose_landmarker_heavy.task'))
    _candidates.append(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'vibe_mocap', 'vibe_mocap', 'pose_landmarker_heavy.task'
    ))
    _candidates.append(os.path.join(os.getcwd(), 'pose_landmarker_heavy.task'))

    model_path = None
    for c in _candidates:
        if os.path.exists(c):
            model_path = os.path.abspath(c)
            break

    if model_path is None:
        print("[vibe] pose_landmarker_heavy.task not found. Searched:")
        for c in _candidates:
            print(f"  {os.path.abspath(c)}")
        print("[vibe] Falling back to mock mode.")
        return None

    try:
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options      = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO
        )
        detector = mp_vision.PoseLandmarker.create_from_options(options)
        print(f"[vibe] MediaPipe detector loaded from: {model_path}")
        return detector
    except Exception as e:
        print(f"[vibe] Failed to load MediaPipe detector: {e}")
        return None


# ── T2.6 — draw_skeleton ─────────────────────────────────────────────────────
def draw_skeleton(frame, norm_landmarks):
    """
    Draw skeleton lines and joint dots on a BGR frame using normalised 2D landmarks.

    Mutates frame in place and also returns it for chaining.
    """
    h, w = frame.shape[:2]

    for (start_idx, end_idx) in POSE_CONNECTIONS:
        sx = int(norm_landmarks[start_idx].x * w)
        sy = int(norm_landmarks[start_idx].y * h)
        ex = int(norm_landmarks[end_idx].x * w)
        ey = int(norm_landmarks[end_idx].y * h)
        cv2.line(frame, (sx, sy), (ex, ey), (255, 255, 255), 2)

    for lm in norm_landmarks:
        cx = int(lm.x * w)
        cy = int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    return frame


# ── T2.7 — detect_posture (KEY function — produces LEFT panel content) ────────
def detect_posture(detector, frame):
    """
    Run posture detection on a BGR frame.

    Always returns (label_str, annotated_bgr_frame).
    In mock mode (detector=None), overlays placeholder text.
    In real mode, draws skeleton + posture label.
    """
    # ── Mock mode guard ──────────────────────────────────────────────────────
    if detector is None:
        overlay = frame.copy()
        cv2.putText(overlay, "VIBE MOCK MODE",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 215, 255), 3, cv2.LINE_AA)
        cv2.putText(overlay, "MediaPipe not available",
                    (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 100), 2, cv2.LINE_AA)
        return "Mock Mode", overlay

    # ── Real mode ────────────────────────────────────────────────────────────
    try:
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts       = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        result   = detector.detect_for_video(mp_image, ts)
    except Exception as e:
        # If MediaPipe crashes mid-use, degrade gracefully
        overlay = frame.copy()
        cv2.putText(overlay, f"VIBE ERR: {str(e)[:40]}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)
        return "Unknown", overlay

    if not result.pose_world_landmarks:
        return "Unknown", frame

    # Extract 3D world landmarks for classification
    pts   = [[lm.x, lm.y, lm.z] for lm in result.pose_world_landmarks[0]]
    label = classify_pose(pts)

    # Draw skeleton using 2D normalised landmarks
    if result.pose_landmarks:
        draw_skeleton(frame, result.pose_landmarks[0])

    # Overlay posture label
    cv2.putText(frame, f"Pose: {label}",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)

    return label, frame
