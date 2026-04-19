"""
G1 Mission Control — Flask Backend
Left panel: Webcam + ROMP skeleton
Right panel: MuJoCo G1 robot via GMR's RobotMotionViewer

SETUP:
  pip install flask opencv-python numpy romp mujoco
  cd GMR && pip install -e . && cd ..

RUN:
  python app.py
  Open http://localhost:5000

CONFIGURATION:
  Change CAMERA_SOURCE below (0 = laptop webcam, or DroidCam URL)
  Change USE_PHONE_CAMERA to True if using a phone held vertically
"""

from flask import Flask, render_template, Response, jsonify
import cv2
import threading
import time
import os
import sys
import numpy as np
from collections import deque

# ── CONFIGURATION — Change these for your setup ──
CAMERA_SOURCE = 0
USE_PHONE_CAMERA = False
ROBOT_NAME = "unitree_g1"

# Add GMR to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'GMR'))

# ── Import pipeline ──
try:
    from run import ROMPGMRPipeline, CameraStream
except ImportError as e:
    print(f"\n[FATAL] Cannot import from run.py: {e}")
    print("Make sure run.py is in the same folder as app.py\n")
    sys.exit(1)

# ── Check dependencies ──
ROMP_OK = False
try:
    import romp
    ROMP_OK = True
except ImportError:
    print("[WARN] ROMP not installed → pip install romp")

GMR_OK = False
try:
    from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting
    GMR_OK = True
except ImportError:
    print("[WARN] GMR not installed → cd GMR && pip install -e .")

VIEWER_OK = False
try:
    from general_motion_retargeting.robot_motion_viewer import RobotMotionViewer
    VIEWER_OK = True
except ImportError:
    print("[WARN] RobotMotionViewer not available")

MUJOCO_OK = False
try:
    import mujoco
    MUJOCO_OK = True
except ImportError:
    print("[WARN] MuJoCo not installed → pip install mujoco")


app = Flask(__name__)

# ── Global State ──
inference_running = False
lock = threading.Lock()

latest_webcam_frame = None
latest_robot_frame = None

metrics = {
    "fps": 0,
    "inference_ms": 0,
    "retarget_ms": 0,
    "render_ms": 0,
    "person_detected": False,
    "status": "idle",
    "frame_count": 0,
}
fps_buffer = deque(maxlen=30)


def run_inference_loop():
    global latest_webcam_frame, latest_robot_frame, inference_running, metrics

    # ══════════════════════════════════
    #  1. INIT PIPELINE
    # ══════════════════════════════════
    metrics["status"] = "loading models..."
    print("\n[INIT] Loading ROMP + GMR pipeline...")

    pipeline = None
    try:
        pipeline = ROMPGMRPipeline(ROBOT_NAME)
        print("[INIT] Pipeline loaded.")
    except Exception as e:
        metrics["status"] = f"pipeline error: {e}"
        print(f"[ERROR] Pipeline init failed: {e}")
        import traceback
        traceback.print_exc()

    # ══════════════════════════════════
    #  2. INIT CAMERA
    # ══════════════════════════════════
    print(f"[INIT] Opening camera (source={CAMERA_SOURCE})...")
    cap = CameraStream(CAMERA_SOURCE)
    if not cap.isOpened():
        fallback = 1 if CAMERA_SOURCE == 0 else 0
        print(f"[WARN] Camera {CAMERA_SOURCE} failed. Trying {fallback}...")
        cap = CameraStream(fallback)
        if not cap.isOpened():
            metrics["status"] = "error: no camera"
            print("[FATAL] No camera found.")
            return
    print("[INIT] Camera opened.")

    # ══════════════════════════════════
    #  3. INIT MUJOCO VIEWER
    # ══════════════════════════════════
    viewer = None
    mj_renderer = None
    mj_model = None
    mj_data = None

    if VIEWER_OK and MUJOCO_OK and pipeline is not None and pipeline.retargeter is not None:
        try:
            viewer = RobotMotionViewer(ROBOT_NAME)
            print(dir(viewer))
            mj_model = viewer.model
            mj_data = viewer.data
            mj_renderer = mujoco.Renderer(mj_model, 640, 480)
            print("[INIT] G1 viewer + offscreen renderer ready")
            print(f"       nq={mj_model.nq}  nv={mj_model.nv}")
        except Exception as e:
            print(f"[WARN] Viewer init failed: {e}")
            print(dir(viewer))
            import traceback
            traceback.print_exc()
            viewer = None

    # Fallback: try loading XML directly
    if mj_renderer is None and MUJOCO_OK:
        print("[INIT] Trying direct XML loading as fallback...")
        for root, dirs, files in os.walk("."):
            for f in files:
                if "g1" in f.lower() and f.endswith(".xml") and "mocap" in f.lower():
                    xml_path = os.path.join(root, f)
                    try:
                        mj_model = mujoco.MjModel.from_xml_path(xml_path)
                        mj_data = mujoco.MjData(mj_model)
                        mj_renderer = mujoco.Renderer(mj_model, 640, 480)
                        print(f"[INIT] Fallback renderer: {xml_path}")
                        break
                    except Exception as e:
                        print(f"[WARN] Failed to load {xml_path}: {e}")
            if mj_renderer is not None:
                break

    if mj_renderer is None:
        print("[WARN] No MuJoCo renderer available. Right panel will be empty.")

    metrics["status"] = "ready"
    print("\n[READY] Open http://localhost:5000 and click START\n")

    prev_time = time.time()
    prev_qpos = None

    # ══════════════════════════════════
    #  MAIN LOOP
    # ══════════════════════════════════
    while True:
        with lock:
            if not inference_running:
                time.sleep(0.05)
                continue

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        if USE_PHONE_CAMERA:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        webcam_display = frame.copy()

        # ── ROMP Pose Estimation ──
        human_motion_dict = None
        if pipeline is not None and pipeline.romp_model is not None:
            t_infer = time.time()
            try:
                human_motion_dict = pipeline.process_frame(frame)
            except Exception as e:
                if metrics["frame_count"] % 200 == 0:
                    print(f"[WARN] ROMP: {e}")
            metrics["inference_ms"] = round((time.time() - t_infer) * 1000, 1)

        metrics["person_detected"] = human_motion_dict is not None

        # ── GMR Retargeting + MuJoCo Render ──
        if human_motion_dict is not None and pipeline is not None and pipeline.retargeter is not None:
            t_retarget = time.time()
            robot_qpos = None
            try:
                robot_qpos = pipeline.retargeter.retarget(
                    human_motion_dict, offset_to_ground=True
                )
            except Exception as e:
                if metrics["frame_count"] % 200 == 0:
                    print(f"[WARN] Retarget: {e}")

            if robot_qpos is not None:
                metrics["retarget_ms"] = round((time.time() - t_retarget) * 1000, 1)

                # EMA smoothing
                alpha = 0.25
                if prev_qpos is not None:
                    q1 = prev_qpos[3:7]
                    q2 = robot_qpos[3:7]
                    if np.dot(q1, q2) < 0:
                        robot_qpos[3:7] = -q2
                    prev_qpos = alpha * robot_qpos + (1 - alpha) * prev_qpos
                    prev_qpos[3:7] /= np.linalg.norm(prev_qpos[3:7])
                    robot_qpos = prev_qpos.copy()
                else:
                    prev_qpos = robot_qpos.copy()

                # Update GMR viewer state
                if viewer is not None:
                    try:
                        root_pos = robot_qpos[:3].copy()
                        root_pos[2] -= 0.06
                        root_rot = robot_qpos[3:7]
                        dof_pos = robot_qpos[7:]
                        viewer.step(root_pos, root_rot, dof_pos, rate_limit=False)
                    except Exception as e:
                        if metrics["frame_count"] % 200 == 0:
                            print(f"[WARN] Viewer step: {e}")

                # Render robot frame for web
                if mj_renderer is not None:
                    try:
                        t_render = time.time()

                        if viewer is not None:
                            # Always use viewer's live data
                            mj_renderer.update_scene(viewer.data)
                        else:
                            n = min(len(robot_qpos), mj_model.nq)
                            mj_data.qpos[:n] = robot_qpos[:n]
                            mujoco.mj_forward(mj_model, mj_data)
                            mj_renderer.update_scene(mj_data)
                        mj_renderer.update_scene(mj_data)
                        robot_rgb = mj_renderer.render()
                        robot_bgr = cv2.cvtColor(robot_rgb, cv2.COLOR_RGB2BGR)

                        cv2.putText(robot_bgr, "G1 Active", (10, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(robot_bgr, f"FPS: {metrics['fps']}", (10, 470),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                        _, buf = cv2.imencode(".jpg", robot_bgr,
                                              [cv2.IMWRITE_JPEG_QUALITY, 80])
                        latest_robot_frame = buf.tobytes()
                        metrics["render_ms"] = round(
                            (time.time() - t_render) * 1000, 1)
                    except Exception as e:
                        if metrics["frame_count"] % 200 == 0:
                            print(f"[WARN] Render: {e}")

            cv2.putText(webcam_display, "Tracking Active", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(webcam_display, "No Person Detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # ── FPS ──
        dt = time.time() - prev_time
        prev_time = time.time()
        fps_buffer.append(1.0 / max(dt, 0.001))
        metrics["fps"] = round(float(np.mean(fps_buffer)), 1)
        metrics["frame_count"] += 1
        metrics["status"] = "running"

        # ── Encode webcam for left panel ──
        _, buf = cv2.imencode(".jpg", webcam_display,
                              [cv2.IMWRITE_JPEG_QUALITY, 80])
        latest_webcam_frame = buf.tobytes()


def gen_stream(stream_type="webcam"):
    """MJPEG generator for either webcam or robot stream."""
    while True:
        if stream_type == "webcam":
            frame = latest_webcam_frame
        else:
            frame = latest_robot_frame
        if frame:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        else:
            time.sleep(0.1)
        time.sleep(0.033)


# ══════════════════════════════════════
#  FLASK ROUTES
# ══════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_stream("webcam"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/robot_feed")
def robot_feed():
    return Response(gen_stream("robot"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/toggle_inference")
def toggle_inference():
    global inference_running
    with lock:
        inference_running = not inference_running
    return jsonify({
        "status": "running" if inference_running else "stopped",
        "inference_running": inference_running,
    })

@app.route("/metrics")
def get_metrics():
    return jsonify(metrics)

@app.route("/health")
def health():
    return jsonify({
        "romp": ROMP_OK,
        "gmr": GMR_OK,
        "viewer": VIEWER_OK,
        "mujoco": MUJOCO_OK,
    })


# ══════════════════════════════════════
#  MAIN
# ══════════════════════════════════════

if __name__ == "__main__":
    print("")
    print("=" * 55)
    print("  G1 MISSION CONTROL")
    print("=" * 55)
    print(f"  ROMP:     {'OK' if ROMP_OK else 'MISSING'}")
    print(f"  GMR:      {'OK' if GMR_OK else 'MISSING'}")
    print(f"  Viewer:   {'OK' if VIEWER_OK else 'MISSING'}")
    print(f"  MuJoCo:   {'OK' if MUJOCO_OK else 'MISSING'}")
    print(f"  Camera:   {CAMERA_SOURCE}")
    print(f"  Phone:    {USE_PHONE_CAMERA}")
    print("")
    print("  http://localhost:5000")
    print("  http://localhost:5000/health")
    print("=" * 55)
    print("")

    threading.Thread(target=run_inference_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False)