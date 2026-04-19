"""
config.py — All tunables for G1 Mission Control webapp.
Do NOT hardcode these values elsewhere. Import from config.

Target PC:  Prateek's Linux laptop
            Python env: /home/prateek/dhakkan/dl_hackathon/AG/romp_gmr_env/bin/python
            Camera: Phone via USB — adb forward tcp:4747 tcp:4747
Run:        cd webapp && python app.py   (no extra config needed)
"""
import os

# ═══════════ TARGET PC DEFAULTS ═══════════
# Pre-configured for Prateek's Linux laptop.
# adb forward tcp:4747 tcp:4747  →  then just: cd webapp && python app.py
CAMERA_SOURCE = "http://localhost:4747/video"   # DroidCam via ADB
USE_PHONE_CAMERA = True                          # Phone held vertically → rotate 90° CW
ROBOT_NAME = "unitree_g1"                        # GMR robot identifier

# ═══════════ WEBAPP SETTINGS ═══════════
VIBE_ENABLED = True
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
JPEG_QUALITY = 85
MJPEG_SLEEP = 0.016                  # ~60fps max stream rate
METRICS_POLL_MS = 300                # Frontend JS polling interval (ms)
EMA_ALPHA = 0.4                      # Jitter smoothing (0.1=smooth/laggy, 1.0=raw/jittery)
FLOOR_OFFSET = 0.06                  # Lower robot by 6cm to fix visual float
RENDER_WIDTH = 640
RENDER_HEIGHT = 480

# ═══════════ AUTO-DETECT MOCK MODE ═══════════
# If heavy dependencies are missing, we're developing locally — use mock frames.
# This flag is checked in app.py, pipeline.py, and vibe.py.
MOCK_MODE = False

try:
    import romp  # noqa: F401
except ImportError:
    MOCK_MODE = True

try:
    import mujoco  # noqa: F401
except ImportError:
    MOCK_MODE = True

if __name__ == "__main__":
    print(f"Config OK — MOCK_MODE={MOCK_MODE}")
    print(f"  CAMERA_SOURCE = {CAMERA_SOURCE}")
    print(f"  ROBOT_NAME    = {ROBOT_NAME}")
    print(f"  FLASK_PORT    = {FLASK_PORT}")
