"""
pipeline.py — ROMP→GMR motion retargeting pipeline for G1 Mission Control.

All heavyweight imports are wrapped in try/except.
This module works on ANY machine (mock mode if deps missing).

sys.path is configured here so ROMP and GMR are discovered automatically
when this file runs inside AGM/webapp/ on the target Linux machine.
"""
import cv2
import time
import numpy as np
import threading
import os
import sys

# ── sys.path: let Python find AGM-level packages (ROMP, GMR) ─────────────────
_HERE   = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
sys.path.insert(0, _PARENT)
sys.path.insert(0, os.path.join(_PARENT, 'GMR'))

# ── Optional heavy imports ────────────────────────────────────────────────────
TORCH_OK   = False
ROMP_OK    = False
GMR_OK     = False
MUJOCO_OK  = False

try:
    import torch  # noqa: F401
    TORCH_OK = True
except ImportError:
    print("[pipeline] torch not found — ROMP will be unavailable")

try:
    from scipy.spatial.transform import Rotation as R
    _R_OK = True
except ImportError:
    print("[pipeline] scipy not found — pose math will be unavailable")
    _R_OK = False

try:
    import romp  # noqa: F401
    ROMP_OK = True
    print("[pipeline] ROMP: OK")
except ImportError:
    print("[pipeline] ROMP not installed — running in mock mode")

try:
    from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting
    GMR_OK = True
    print("[pipeline] GMR:  OK")
except ImportError:
    GeneralMotionRetargeting = None
    print("[pipeline] GMR not found — running in mock mode")

try:
    import mujoco  # noqa: F401
    MUJOCO_OK = True
    print("[pipeline] MuJoCo: OK")
except ImportError:
    print("[pipeline] MuJoCo not installed — running in mock mode")

# ── SMPL Constants ────────────────────────────────────────────────────────────
SMPL_JOINT_NAMES = [
    'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2',
    'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar',
    'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow',
    'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'
]

SMPL_TO_GMR_NAMES = {
    'Pelvis':     'pelvis',
    'L_Hip':      'left_hip',
    'R_Hip':      'right_hip',
    'L_Knee':     'left_knee',
    'R_Knee':     'right_knee',
    'L_Ankle':    'left_foot',
    'R_Ankle':    'right_foot',
    'Spine3':     'spine3',
    'L_Shoulder': 'left_shoulder',
    'R_Shoulder': 'right_shoulder',
    'L_Elbow':    'left_elbow',
    'R_Elbow':    'right_elbow',
    'L_Wrist':    'left_wrist',
    'R_Wrist':    'right_wrist',
}

# Kinematic parent tree (SMPL 24-joint topology)
SMPL_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21
]

# ─────────────────────────────────────────────────────────────────────────────
#  T1.2 — CameraStream (threaded, low-latency capture)
# ─────────────────────────────────────────────────────────────────────────────
class CameraStream:
    """
    Threaded video capture that continuously pulls frames in the background,
    eliminating OpenCV's internal network buffer lag.
    """
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            self.grabbed, self.frame = self.stream.read()

    def read(self):
        return self.grabbed, self.frame

    def release(self):
        self.stopped = True
        self.thread.join(timeout=2)
        self.stream.release()

    def isOpened(self):
        return self.stream.isOpened()


# ─────────────────────────────────────────────────────────────────────────────
#  T1.3 — process_frame: ROMP inference → joint dict for GMR
# ─────────────────────────────────────────────────────────────────────────────
def process_frame(frame, romp_model):
    """
    Run ROMP on a BGR frame and convert to a GMR-compatible joint dict.

    Returns dict[str → (pos_3d, rot_4d)] or None if no person detected.
    Guard: returns None immediately if romp_model is None (mock mode).
    """
    if romp_model is None or not _R_OK:
        return None

    outputs = romp_model(frame)
    if outputs is None or len(outputs.get('smpl_thetas', [])) == 0:
        return None

    smpl_thetas = outputs['smpl_thetas'][0]   # (72,)
    cam_trans   = outputs['cam_trans'][0]      # (3,)
    joints_3d   = outputs.get('joints', [None])[0]

    axis_angles = np.array(smpl_thetas).reshape(-1, 3)

    # Forward kinematics: accumulate global rotations down the kinematic tree
    global_rots = []
    for i in range(24):
        if i == 0:
            rot = R.from_rotvec(axis_angles[0])
        else:
            rot = global_rots[SMPL_PARENTS[i]] * R.from_rotvec(axis_angles[i])
        global_rots.append(rot)

    # Coordinate correction: OpenCV Y-down → MuJoCo Z-up
    rotation_matrix = np.array([
        [1,  0,  0],
        [0,  0,  1],
        [0, -1,  0]
    ])
    gmr_rot_correction = R.from_matrix(rotation_matrix)

    human_motion_dict = {}
    for i, joint_name in enumerate(SMPL_JOINT_NAMES):
        if i >= 24:
            break

        final_rot = (gmr_rot_correction * global_rots[i]).as_quat(scalar_first=True)

        if joints_3d is not None and i < len(joints_3d):
            translation = joints_3d[i]
        else:
            translation = cam_trans if i == 0 else np.zeros(3)

        final_pos = translation @ rotation_matrix.T
        gmr_name  = SMPL_TO_GMR_NAMES.get(joint_name, joint_name.lower())
        human_motion_dict[gmr_name] = (final_pos, final_rot)

    return human_motion_dict


# ─────────────────────────────────────────────────────────────────────────────
#  T1.4 — smooth_qpos: EMA smoothing with quaternion sign-flip fix
# ─────────────────────────────────────────────────────────────────────────────
def smooth_qpos(raw_qpos, prev_qpos, alpha):
    """
    Blend raw_qpos toward prev_qpos using EMA (alpha=recent weight).
    Fixes quaternion sign flipping before interpolation.
    Returns (smoothed_qpos, updated_prev_qpos).
    """
    robot_qpos = raw_qpos.copy()

    # Fix sign ambiguity of root quaternion
    q1 = prev_qpos[3:7]
    q2 = robot_qpos[3:7]
    if np.dot(q1, q2) < 0:
        robot_qpos[3:7] = -q2

    # EMA blend
    smoothed = alpha * robot_qpos + (1.0 - alpha) * prev_qpos
    # Re-normalize root quaternion
    smoothed[3:7] /= np.linalg.norm(smoothed[3:7]) + 1e-9

    return smoothed, smoothed.copy()


# ─────────────────────────────────────────────────────────────────────────────
#  T1.5 — init_romp
# ─────────────────────────────────────────────────────────────────────────────
def init_romp():
    """Initialise ROMP model. Returns None in mock mode."""
    if not ROMP_OK:
        print("[pipeline] init_romp: ROMP unavailable — returning None (mock mode)")
        return None
    try:
        settings = romp.main.default_settings
        settings.show_largest_person_only = True
        model = romp.ROMP(settings)
        print("[pipeline] ROMP model loaded OK")
        return model
    except Exception as e:
        print(f"[pipeline] init_romp failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  T1.6 — init_gmr
# ─────────────────────────────────────────────────────────────────────────────
def init_gmr(robot_name):
    """Initialise GMR retargeter. Returns None in mock mode."""
    if not GMR_OK:
        print("[pipeline] init_gmr: GMR unavailable — returning None (mock mode)")
        return None
    try:
        retargeter = GeneralMotionRetargeting(
            "smplx", robot_name,
            use_velocity_limit=True,
            damping=0.02
        )
        print(f"[pipeline] GMR retargeter loaded for '{robot_name}'")
        return retargeter
    except Exception as e:
        print(f"[pipeline] init_gmr failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  T1.7 — init_mujoco_renderer (offscreen — no OS window)
# ─────────────────────────────────────────────────────────────────────────────
def init_mujoco_renderer(robot_name, width, height):
    """
    Initialise MuJoCo offscreen renderer.
    Returns (model, data, renderer) or (None, None, None) in mock mode.
    Does NOT open any OS window.
    """
    if not MUJOCO_OK or not GMR_OK:
        print("[pipeline] init_mujoco_renderer: deps unavailable — returning None (mock mode)")
        return None, None, None
    try:
        from general_motion_retargeting.params import ROBOT_XML_DICT
        xml_path = str(ROBOT_XML_DICT[robot_name])
        model    = mujoco.MjModel.from_xml_path(xml_path)
        data     = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height, width)
        print(f"[pipeline] MuJoCo renderer ready ({width}×{height})")
        return model, data, renderer
    except Exception as e:
        print(f"[pipeline] init_mujoco_renderer failed: {e}")
        return None, None, None


# ─────────────────────────────────────────────────────────────────────────────
#  T1.8 — generate_mock_frame (cv2 + numpy only — works on Windows)
# ─────────────────────────────────────────────────────────────────────────────
def generate_mock_frame(width, height, text, color=(100, 200, 100)):
    """
    Generate a dark placeholder frame for mock mode.
    Only uses cv2 + numpy — works on any machine without pipeline deps.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8) + 15

    # Subtle border
    cv2.rectangle(frame, (10, 10), (width - 10, height - 10), (40, 40, 50), 1)

    # Diagonal grid lines (very subtle)
    for x in range(0, width, 80):
        cv2.line(frame, (x, 0), (x, height), (20, 20, 28), 1)
    for y in range(0, height, 80):
        cv2.line(frame, (0, y), (width, y), (20, 20, 28), 1)

    # Main label
    font       = cv2.FONT_HERSHEY_SIMPLEX
    text_size  = cv2.getTextSize(text, font, 0.7, 2)[0]
    text_x     = (width  - text_size[0]) // 2
    text_y     = (height - text_size[1]) // 2

    cv2.putText(frame, text, (text_x, text_y),
                font, 0.7, color, 2, cv2.LINE_AA)

    # Subtitle
    sub        = "Deploy to target machine for real inference"
    sub_size   = cv2.getTextSize(sub, font, 0.45, 1)[0]
    sub_x      = (width - sub_size[0]) // 2
    cv2.putText(frame, sub, (sub_x, text_y + 36),
                font, 0.45, (70, 70, 90), 1, cv2.LINE_AA)

    return frame
