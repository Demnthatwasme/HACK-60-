# G1 Mission Control — Webapp Technical Documentation

> **File location**: `AGM/webapp/WEBAPP.md`
> **Last updated**: 2026-04-19
> **Scope**: Everything you need to understand, configure, and deploy the webapp on a new machine.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Repository Layout](#2-repository-layout)
3. [Architecture](#3-architecture)
   - 3.1 [Thread Model](#31-thread-model)
   - 3.2 [Data Flow — Real Mode](#32-data-flow--real-mode)
   - 3.3 [Data Flow — Mock Mode](#33-data-flow--mock-mode)
   - 3.4 [Panel Assignment](#34-panel-assignment)
4. [Files Reference](#4-files-reference)
   - 4.1 [config.py](#41-configpy)
   - 4.2 [pipeline.py](#42-pipelinepy)
   - 4.3 [vibe.py](#43-vibepy)
   - 4.4 [app.py](#44-apppy)
   - 4.5 [static/css/dashboard.css](#45-staticcssdashboardcss)
   - 4.6 [templates/index.html](#46-templatesindexhtml)
5. [HTTP API Reference](#5-http-api-reference)
6. [Configuration Reference](#6-configuration-reference)
7. [Mock Mode Design](#7-mock-mode-design)
8. [Dependency Matrix](#8-dependency-matrix)
9. [Installation & First Run](#9-installation--first-run)
   - 9.1 [Windows Dev Machine (Mock Mode)](#91-windows-dev-machine-mock-mode)
   - 9.2 [Linux Target Machine (Live Mode)](#92-linux-target-machine-live-mode)
10. [Camera Setup](#10-camera-setup)
11. [Troubleshooting](#11-troubleshooting)
12. [Design System Quick Reference](#12-design-system-quick-reference)

---

## 1. Overview

**G1 Mission Control** is a self-contained Flask web application that replaces two previously separate Python scripts (`webcam_to_gmr.py` and `vibe_mocap/mocap.py`) with a unified pipeline + web dashboard.

**What it does (live mode on Linux target):**

```
Phone Camera (DroidCam/ADB)
        │
        ▼
  ROMP Pose Estimation  ──→  SMPL 3D joint parameters (72 values)
        │
        ▼
  GMR Retargeting       ──→  Unitree G1 joint angles (qpos vector)
        │
        ▼
  MuJoCo Offscreen      ──→  Rendered RGB frame of the G1 robot
  Renderer                   (RIGHT browser panel)
        │
        └────────────────────→  Also feeds MediaPipe VIBE thread
                                    │
                                    ▼
                              Posture Classification
                              Skeleton Overlay
                              (LEFT browser panel)
```

**On any other machine** (Windows, macOS, any Linux without the model deps), the app starts in **Mock Mode** — the dashboard loads fully, streams show placeholder frames, and all JS/metrics still work.

---

## 2. Repository Layout

```
AGM/                              ← root of the project
├── webcam_to_gmr.py              ← legacy standalone CLI (untouched)
├── abc.py                        ← earlier prototype (untouched)
├── setup_env.sh                  ← conda/pip env setup for Linux target
├── GMR/                          ← git submodule — General Motion Retargeting
├── ROMP/                         ← git submodule — ROMP pose estimator
├── vibe_mocap/
│   └── vibe_mocap/
│       ├── mocap.py              ← legacy posture script (untouched)
│       └── pose_landmarker_heavy.task   ← MediaPipe model (30 MB)
│
└── webapp/                       ← THIS FOLDER — entire web app
    ├── WEBAPP.md                 ← you are reading this
    ├── app.py                    ← Flask server + background threads
    ├── pipeline.py               ← ROMP → GMR logic + mock frame generator
    ├── vibe.py                   ← MediaPipe classifier + mock fallback
    ├── config.py                 ← all tunables (edit this for your machine)
    ├── static/
    │   └── css/
    │       └── dashboard.css     ← full dark-mode design system
    └── templates/
        └── index.html            ← dashboard HTML + inline JavaScript
```

> **Rule**: Never import from `webcam_to_gmr.py` or `mocap.py` directly.
> All shared logic was ported verbatim into `pipeline.py` and `vibe.py`.

---

## 3. Architecture

### 3.1 Thread Model

The Flask process runs exactly **3 threads**:

| Thread | Function | Produces |
|---|---|---|
| Main | Flask HTTP server | Handles all HTTP requests |
| Thread-1 | `inference_loop()` | `latest_robot_frame` (JPEG bytes) |
| Thread-2 | `vibe_classifier_loop()` | `latest_vibe_frame` (JPEG bytes) |

All three threads share state via module-level globals. Writes to `inference_running` use a `threading.Lock`. Reads of frame buffers (`latest_*_frame`, `latest_webcam_raw`) are unguarded single-assignment — GIL is sufficient because they are updated atomically (single object pointer swap).

### 3.2 Data Flow — Real Mode

```
┌────────────────────────────────────────────────────────────────────────┐
│ Thread-1: inference_loop()                                             │
│                                                                        │
│  CameraStream.read()                                                   │
│       │  (threaded background capture — no buffer lag)                 │
│       ▼                                                                │
│  cv2.rotate() if USE_PHONE_CAMERA                                      │
│       │                                                                │
│  ┌────┴──────────────────────────────────────┐                         │
│  │ latest_webcam_raw = frame.copy()          │ ◄─── shared globals    │
│  └────┬──────────────────────────────────────┘                         │
│       │                                                                │
│       ▼                                                                │
│  pipeline.process_frame(frame, romp_model)                             │
│       │  Returns dict[joint_name → (pos_3d, rot_quat)] or None        │
│       ▼                                                                │
│  retargeter.retarget(human_motion_dict)                                │
│       │  Returns numpy qpos vector (3+4+N_dof values)                 │
│       ▼                                                                │
│  pipeline.smooth_qpos(raw, prev, EMA_ALPHA)                            │
│       │  EMA blend + quaternion sign-flip fix                          │
│       ▼                                                                │
│  mujoco.mj_forward() → renderer.update_scene() → renderer.render()    │
│       │  Single update_scene() call (avoids double-render bug)        │
│       ▼                                                                │
│  cv2.imencode(".jpg") → latest_robot_frame (bytes)                    │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│ Thread-2: vibe_classifier_loop()                                       │
│                                                                        │
│  Reads latest_webcam_raw (numpy) ──────────────────────────────────── │
│       │  (snapshot copy to avoid race with Thread-1 overwrite)        │
│       ▼                                                                │
│  vibe.detect_posture(detector, frame)                                  │
│       │  MediaPipe VIDEO mode → world + 2D landmarks                  │
│       │  → vibe.classify_pose() → label string                        │
│       │  → vibe.draw_skeleton() → joint dots + bone lines             │
│       │  → cv2.putText(label)                                          │
│       ▼                                                                │
│  current_posture = label                                               │
│  cv2.imencode(".jpg") → latest_vibe_frame (bytes)                     │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│ Main Thread: Flask                                                     │
│                                                                        │
│  GET /vibe_feed  → gen_stream("vibe")  → reads latest_vibe_frame      │
│  GET /robot_feed → gen_stream("robot") → reads latest_robot_frame     │
│  (all other routes are fast JSON endpoints — no blocking)              │
└────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Data Flow — Mock Mode

When `MOCK_MODE = True` (any missing heavy dep triggers this):

```
inference_loop():
  frame = mock_webcam.copy()       ← static dark frame, generated once at startup
  latest_webcam_raw = frame        ← VIBE thread still gets something to annotate
  latest_robot_frame = encode(mock_robot)  ← static placeholder JPEG

vibe_classifier_loop():
  detect_posture(detector=None, frame)
      └─ returns ("Mock Mode", frame_with_text_overlay)
  latest_vibe_frame = encode(annotated_mock_frame)
```

Both MJPEG streams stay active — just serving placeholder frames. All metrics, routes, and JS still work identically.

### 3.4 Panel Assignment

| Browser Panel | Flask Route | Frame Source | Content |
|---|---|---|---|
| **LEFT** | `/vibe_feed` | `latest_vibe_frame` | MediaPipe skeleton + posture label overlaid on webcam |
| **RIGHT** | `/robot_feed` | `latest_robot_frame` | MuJoCo offscreen render of retargeted Unitree G1 |

> ⚠️ There is **no raw webcam stream** exposed to the browser. The raw frame is only shared internally as `latest_webcam_raw` (a numpy array in memory).

---

## 4. Files Reference

### 4.1 `config.py`

**Single source of truth for all configuration.** Never hardcode values in other files.

```python
# Key tunables:
CAMERA_SOURCE    = "http://localhost:4747/video"  # DroidCam ADB URL
USE_PHONE_CAMERA = True           # Rotate 90° CW (phone held vertically)
ROBOT_NAME       = "unitree_g1"   # Must match a key in GMR's ROBOT_XML_DICT
FLASK_PORT       = 5000
JPEG_QUALITY     = 85             # 0–100, affects bandwidth vs. quality
MJPEG_SLEEP      = 0.016          # 16ms = 60fps stream cap per client
METRICS_POLL_MS  = 300            # JS polling interval (milliseconds)
EMA_ALPHA        = 0.4            # Smoothing: 0.1=laggy, 1.0=jittery
FLOOR_OFFSET     = 0.06           # Lowers root by 6cm to prevent visual float
RENDER_WIDTH     = 640
RENDER_HEIGHT    = 480
MOCK_MODE        = False/True     # Auto-detected — do not set manually
```

**MOCK_MODE detection logic:**
```python
MOCK_MODE = False
try:
    import romp
except ImportError:
    MOCK_MODE = True
try:
    import mujoco
except ImportError:
    MOCK_MODE = True
```

Both checks run independently — if either fails, `MOCK_MODE` is set to `True`.

---

### 4.2 `pipeline.py`

**Responsibilities**: Camera capture, ROMP inference, forward kinematics, GMR retargeting, MuJoCo rendering, mock frame generation.

**Module-level flags** (set after import attempt):

| Flag | Default | Meaning |
|---|---|---|
| `ROMP_OK` | `False` | `import romp` succeeded |
| `GMR_OK` | `False` | `import general_motion_retargeting` succeeded |
| `MUJOCO_OK` | `False` | `import mujoco` succeeded |
| `TORCH_OK` | `False` | `import torch` succeeded |

**Public API:**

```python
class CameraStream(src)
    # Threaded video capture. CAP_PROP_BUFFERSIZE=1, daemon thread.
    .read()    → (bool grabbed, np.ndarray frame)
    .isOpened() → bool
    .release()  → None

def process_frame(frame, romp_model) → dict | None
    # Runs ROMP, forward kinematics, coord correction.
    # Returns: {"joint_name": (pos_3d np.ndarray, rot_quat np.ndarray)}
    # Returns None if romp_model is None OR no person detected.

def smooth_qpos(raw_qpos, prev_qpos, alpha) → (smoothed, new_prev)
    # EMA blend with quaternion sign-flip fix.

def init_romp() → romp.ROMP | None
def init_gmr(robot_name) → GeneralMotionRetargeting | None
def init_mujoco_renderer(robot_name, width, height) → (model, data, renderer) | (None, None, None)

def generate_mock_frame(width, height, text, color=(100,200,100)) → np.ndarray
    # Returns dark BGR frame with centered label text.
    # Only uses cv2 + numpy — no pipeline deps required.
```

**Coordinate correction** (OpenCV → MuJoCo):
```python
rotation_matrix = np.array([
    [1,  0,  0],
    [0,  0,  1],
    [0, -1,  0]
])
# Applied to both translation vectors and rotation quaternions.
```

**sys.path setup** (at module top, before any local imports):
```python
_HERE   = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)          # → AGM/
sys.path.insert(0, _PARENT)              # for romp
sys.path.insert(0, os.path.join(_PARENT, 'GMR'))  # for general_motion_retargeting
```

---

### 4.3 `vibe.py`

**Responsibilities**: MediaPipe PoseLandmarker initialisation, 2D skeleton drawing, 3D posture classification.

**Module-level flag:**

| Flag | Default | Meaning |
|---|---|---|
| `MEDIAPIPE_OK` | `False` | `import mediapipe` + task API succeeded |

**Skeleton map**: `POSE_CONNECTIONS` — 33 tuples of `(start_index, end_index)` covering all MediaPipe body landmark connections. Defined at module level.

**Public API:**

```python
def calculate_angle(a, b, c) → float
    # Angle in degrees at point b.

def dist(p1, p2) → float
    # 3D Euclidean distance.

def classify_pose(landmarks: list[list[float]]) → str
    # Input: 33 MediaPipe world landmarks [[x,y,z], ...]
    # Returns one of: "Clap", "Dab Pose", "T-Pose", "Karate Pose",
    #   "Both Hands Up", "Left Hand Up", "Right Hand Up",
    #   "Squats/Crouch", "Sitting", "Leaning Forward",
    #   "Leaning Backward", "Standing"

def init_mediapipe_detector(model_dir=None) → PoseLandmarker | None
    # Searches for pose_landmarker_heavy.task in 3 locations.
    # Always returns None gracefully (never raises).

def draw_skeleton(frame, norm_landmarks) → np.ndarray
    # Draws white bone lines + red joint dots on BGR frame in-place.
    # Uses 2D normalised landmarks from result.pose_landmarks[0].

def detect_posture(detector, frame) → (str, np.ndarray)
    # KEY function — always returns (label, annotated_bgr_frame).
    # If detector is None: overlays "VIBE MOCK MODE" text, returns ("Mock Mode", frame).
    # If detector is real: full MediaPipe inference → classify → draw → return.
```

**MediaPipe model search order** in `init_mediapipe_detector()`:
1. `model_dir` argument (passed as `AGM/vibe_mocap/vibe_mocap/` by `app.py`)
2. `../vibe_mocap/vibe_mocap/pose_landmarker_heavy.task` relative to `vibe.py`
3. `os.getcwd()/pose_landmarker_heavy.task`

**MediaPipe mode**: `RunningMode.VIDEO` — requires monotonic timestamps and is more accurate than LIVE_STREAM. Timestamps computed via:
```python
ts = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
```

---

### 4.4 `app.py`

**Responsibilities**: Flask app, route handlers, background thread management, global state.

**Startup sequence:**
1. `sys.path` configured (same as pipeline.py — required before local imports)
2. All imports (Flask, cv2, config, pipeline, vibe)
3. Global state variables initialised
4. `if __name__ == "__main__"`: print banner → start daemon threads → `app.run()`

**Global shared state:**

```python
inference_running: bool       # toggled by /toggle_inference
lock: threading.Lock          # guards inference_running only
latest_vibe_frame:  bytes     # JPEG — LEFT panel stream
latest_robot_frame: bytes     # JPEG — RIGHT panel stream
latest_webcam_raw:  np.ndarray  # raw BGR — passed to VIBE thread
current_posture:    str       # last classified posture label
metrics:            dict      # see /metrics route
fps_buffer:         deque(30) # rolling window for FPS calculation
```

**Metrics dict schema:**
```json
{
  "fps":             0,
  "inference_ms":    0,
  "retarget_ms":     0,
  "render_ms":       0,
  "person_detected": false,
  "status":          "idle | loading models... | ready | ready (mock) | running | stopped",
  "frame_count":     0,
  "posture":         "Unknown",
  "mock_mode":       true
}
```

**FPS calculation:**
```python
fps_buffer.append(1.0 / max(elapsed, 0.001))
metrics["fps"] = round(float(np.mean(fps_buffer)), 1)
```
Uses exponential moving average window of 30 samples — stable display even at variable frame rates.

**`inference_loop()` flow (simplified):**
```
startup:
  init models → open camera → generate mock frames → set status="ready"

each iteration:
  if not inference_running: sleep(0.05); continue

  frame = camera.read() OR mock_webcam.copy()
  latest_webcam_raw = frame.copy()

  if romp_model:
      human_motion_dict = process_frame(frame, romp_model)
  if human_motion_dict and retargeter:
      robot_qpos = retargeter.retarget(...)
  if robot_qpos:
      robot_qpos = smooth_qpos(...)
  if robot_qpos and mj_renderer:
      mujoco render → latest_robot_frame
  else:
      encode(mock_robot) → latest_robot_frame

  update fps, frame_count, person_detected, posture in metrics
```

**`vibe_classifier_loop()` flow:**
```
startup:
  detector = init_mediapipe_detector(AGM/vibe_mocap/vibe_mocap/)

each iteration (30 Hz):
  if latest_webcam_raw is None: sleep(0.1); continue
  frame = latest_webcam_raw.copy()
  posture, annotated = detect_posture(detector, frame)
  current_posture = posture
  encode(annotated) → latest_vibe_frame
  sleep(0.03)
```

---

### 4.5 `static/css/dashboard.css`

Industrial dark-mode design system. All design tokens defined as CSS custom properties in `:root`.

**Color palette:**

| Variable | Value | Usage |
|---|---|---|
| `--bg-primary` | `#08090c` | Page background |
| `--bg-card` | `#0d0f14` | Card backgrounds |
| `--bg-card-hover` | `#12151c` | Card hover state |
| `--border` | `#1a1d27` | Default borders |
| `--border-active` | `#2a2d3a` | Hover/active borders |
| `--text-primary` | `#e4e6ed` | Body text |
| `--text-secondary` | `#6b7089` | Labels, subtitles |
| `--text-muted` | `#3d4058` | Timestamps, disabled |
| `--accent-green` | `#34d399` | FPS, live status, OK |
| `--accent-red` | `#f87171` | Stop button, errors |
| `--accent-blue` | `#60a5fa` | Step numbers, sim badge |
| `--accent-amber` | `#fbbf24` | Mock banner, frame count |
| `--accent-cyan` | `#22d3ee` | Inference timing |
| `--accent-purple` | `#a78bfa` | VIBE posture display |

**Typography:**
- Headers / UI text: `Outfit` (Google Fonts) — weights 400/600/800/900
- Metrics / code / labels: `JetBrains Mono` (Google Fonts) — weights 400/600/700

**Key layout classes:**

```
.dashboard-grid      grid-template-columns: 1fr 1fr; gap: 20px
.pipeline-info       flex row, spans 2 columns
.stream-card         border 1px solid --border, radius 14px, overflow hidden
.stream-viewport     aspect-ratio 16/9, dark background
.metrics-bar         auto-fit grid, min 160px per card, spans 2 columns
.telemetry           log section, custom scrollbar, spans 2 columns
```

**Animations:**
- `pulse-dot` — green status dot pulses at 2s interval when running
- `fadeIn` — metric cards fade in + slide up 8px on load (0.3s)

---

### 4.6 `templates/index.html`

Single-page dashboard. All JavaScript is inline in a `<script>` block at the bottom.

**DOM IDs used by JavaScript:**

| ID | Element | Purpose |
|---|---|---|
| `statusDot` | `<span>` | Green pulse when running |
| `statusText` | `<span>` | "Idle" / "Running" |
| `statusChip` | `<div>` | Container for dot+text |
| `toggleBtn` | `<button>` | Start/Stop trigger |
| `mockBanner` | `<div>` | Hidden until health check |
| `vibeBadge` | `<span>` | "● Live" badge on left card |
| `simBadge` | `<span>` | "● Sim" badge on right card |
| `vibeFeed` | `<img>` | LEFT panel MJPEG stream |
| `robotFeed` | `<img>` | RIGHT panel MJPEG stream |
| `vibePlaceholder` | `<div>` | Shown when stream is off |
| `simPlaceholder` | `<div>` | Shown when stream is off |
| `metricFps` | `<div>` | Pipeline FPS value |
| `metricInference` | `<div>` | ROMP ms value |
| `metricRetarget` | `<div>` | GMR ms value |
| `metricRender` | `<div>` | MuJoCo ms value |
| `metricPerson` | `<div>` | "Yes"/"No" + color class |
| `metricPosture` | `<div>` | VIBE posture label |
| `metricFrames` | `<div>` | Frame counter |
| `logEntries` | `<div>` | System log container |

**JavaScript functions:**

```javascript
toggleInference()
    // POST-free: GET /toggle_inference → parse JSON → update all UI state.
    // On start: set img.src to stream URLs, show badges, start polling.
    // On stop:  set img.src = "", hide badges, stop polling.

startMetricsPolling() / stopMetricsPolling()
    // setInterval(fetchMetrics, 300) / clearInterval()

fetchMetrics()
    // GET /metrics → update all 7 metric card values + person color class.

addLog(msg)
    // Prepends a timestamped log-line div. Max 30 entries (trims oldest).
    // XSS-safe via escapeHtml().

clearLog()
    // Clears #logEntries innerHTML.

checkHealth()
    // Run on DOMContentLoaded. GET /health → show mock banner if needed
    // + log all dependency statuses.
```

---

## 5. HTTP API Reference

All routes are `GET`. No authentication required.

### `GET /`
Returns the full dashboard HTML page.

**Response**: `text/html`

---

### `GET /vibe_feed`
MJPEG stream — **LEFT panel** content. MediaPipe skeleton + posture label overlaid on webcam frame.

**Response**: `multipart/x-mixed-replace; boundary=frame`

**Frame rate**: Capped at `1 / MJPEG_SLEEP` (default 60fps). Actual rate limited by `vibe_classifier_loop()` (~30 Hz).

---

### `GET /robot_feed`
MJPEG stream — **RIGHT panel** content. MuJoCo offscreen render of retargeted Unitree G1.

**Response**: `multipart/x-mixed-replace; boundary=frame`

---

### `GET /toggle_inference`
Flips the `inference_running` boolean.

**Response**:
```json
{
  "status": "running" | "stopped",
  "inference_running": true | false
}
```

---

### `GET /metrics`
Returns all pipeline performance metrics. Polled by JS every 300ms.

**Response**:
```json
{
  "fps": 23.4,
  "inference_ms": 187.2,
  "retarget_ms": 14.5,
  "render_ms": 6.1,
  "person_detected": true,
  "status": "running",
  "frame_count": 1247,
  "posture": "Standing",
  "mock_mode": false
}
```

---

### `GET /health`
Reports which heavy dependencies successfully loaded at startup.

**Response**:
```json
{
  "romp":      true | false,
  "gmr":       true | false,
  "mujoco":    true | false,
  "vibe":      true | false,
  "mock_mode": true | false
}
```

> On the target Linux machine: all values should be `true` and `mock_mode: false`.
> On a dev machine: all heavy deps `false`, `mock_mode: true`, `vibe` may be `true` if MediaPipe is installed.

---

### `GET /posture`
Latest classified posture label.

**Response**:
```json
{ "posture": "Standing" }
```

---

## 6. Configuration Reference

Edit `AGM/webapp/config.py`. Changes take effect on next `python app.py` restart.

| Constant | Default | Description |
|---|---|---|
| `CAMERA_SOURCE` | `"http://localhost:4747/video"` | DroidCam ADB URL, or `0` for laptop webcam |
| `USE_PHONE_CAMERA` | `True` | Apply 90° CW rotation (phone held vertically) |
| `ROBOT_NAME` | `"unitree_g1"` | Robot key for GMR's `ROBOT_XML_DICT` |
| `VIBE_ENABLED` | `True` | Enable/disable VIBE classifier thread |
| `FLASK_HOST` | `"0.0.0.0"` | Bind to all interfaces (use `"127.0.0.1"` to restrict) |
| `FLASK_PORT` | `5000` | HTTP port |
| `JPEG_QUALITY` | `85` | MJPEG quality 0–100 (affects CPU and bandwidth) |
| `MJPEG_SLEEP` | `0.016` | Seconds between MJPEG frames (~60fps cap) |
| `METRICS_POLL_MS` | `300` | JS polling interval for /metrics |
| `EMA_ALPHA` | `0.4` | Smoothing factor (0.1=heavy smoothing, 1.0=raw) |
| `FLOOR_OFFSET` | `0.06` | Meters to subtract from G1 root Z (visual floor fix) |
| `RENDER_WIDTH` | `640` | MuJoCo offscreen render width (pixels) |
| `RENDER_HEIGHT` | `480` | MuJoCo offscreen render height (pixels) |
| `MOCK_MODE` | auto | **Do not set manually** — auto-detected from imports |

---

## 7. Mock Mode Design

Mock mode exists so the webapp can be developed and tested on **any machine**, not just the Linux target with all heavy dependencies.

### Trigger Condition

`MOCK_MODE` is set to `True` if **either** of these imports fails:
```python
import romp    # ROMP pose estimator
import mujoco  # MuJoCo physics
```

GMR and MediaPipe failures do NOT alone trigger `MOCK_MODE`, but they set their own flags (`GMR_OK`, `MEDIAPIPE_OK`) that cause graceful degradation.

### What Works in Mock Mode

| Feature | Real Mode | Mock Mode |
|---|---|---|
| Dashboard loads | ✅ | ✅ |
| Start/Stop toggle | ✅ | ✅ |
| MJPEG streams | ✅ real frames | ✅ placeholder frames |
| Metrics (FPS) | ✅ real | ✅ real (loop runs) |
| Metrics (timing) | ✅ real ms | ✅ shows 0 |
| Posture label | ✅ real | ✅ "Mock Mode" |
| System log | ✅ | ✅ |
| Mock banner | hidden | ✅ amber banner |
| `/health` endpoint | all true | shows exactly what's missing |

### Placeholder Frame Appearance

```
[dark background, #0f0f0f approx]
[subtle grid lines]
[centered text — e.g. "WEBCAM PLACEHOLDER" in green]
[subtitle — "Deploy to target machine for real inference"]
```

Generated by `pipeline.generate_mock_frame()` using only `cv2` + `numpy` — no pipeline deps.

---

## 8. Dependency Matrix

| Package | Used in | Required for | Install |
|---|---|---|---|
| `flask` | `app.py` | Serving the webapp | `pip install flask` |
| `opencv-python` | All files | Frame capture, encode, draw | `pip install opencv-python` |
| `numpy` | All files | Array math | `pip install numpy` |
| `romp` | `pipeline.py` | Pose estimation | See `ROMP/README.md` |
| `torch` | `pipeline.py` | ROMP backend | `pip install torch` (or CUDA) |
| `scipy` | `pipeline.py` | Rotation math (`Rotation` class) | `pip install scipy` |
| `general_motion_retargeting` | `pipeline.py` | Joint retargeting | `pip install -e GMR/` |
| `mujoco` | `pipeline.py` | Robot simulation | `pip install mujoco` |
| `mediapipe` | `vibe.py` | Posture classification | `pip install mediapipe` |

> **Minimum to run the dashboard in mock mode**: `flask`, `opencv-python-headless`, `numpy`

---

## 9. Installation & First Run

### 9.1 Windows Dev Machine (Mock Mode)

```powershell
# 1. Install minimum dependencies
pip install flask opencv-python-headless numpy

# (Optional) Install MediaPipe for posture classification:
pip install mediapipe

# 2. Start the server
# Find the correct python executable (Windows Store Python recommended):
& "C:\Users\<YOU>\AppData\Local\Microsoft\WindowsApps\python.exe" AGM\webapp\app.py

# 3. Open dashboard
# Navigate to: http://localhost:5000
```

Expected startup output:
```
[pipeline] ROMP not installed — running in mock mode
[pipeline] GMR not found — running in mock mode
[pipeline] MuJoCo not installed — running in mock mode
[vibe] MediaPipe: OK   (or: MediaPipe not installed — mock mode active)

============================================
  G1 MISSION CONTROL -- Web Dashboard
============================================
  Mode:    MOCK
  ROMP:    MISSING
  GMR:     MISSING
  MuJoCo: MISSING
  VIBE:    OK
  Camera:  http://localhost:4747/video
  http://localhost:5000
============================================
```

---

### 9.2 Linux Target Machine (Live Mode)

**Prerequisites already installed** in `romp_gmr_env` by `setup_env.sh`.

```bash
# 1. Connect phone camera
adb forward tcp:4747 tcp:4747

# 2. Activate the conda/pip environment
source /home/prateek/dhakkan/dl_hackathon/AG/romp_gmr_env/bin/activate

# 3. Install Flask (only new dependency not in the original env)
pip install flask

# 4. Run the webapp
cd /home/prateek/dhakkan/dl_hackathon/AG/webapp
python app.py

# 5. Open from any device on the same network
# http://<linux-machine-ip>:5000
```

Expected startup output:
```
[pipeline] ROMP: OK
[pipeline] GMR:  OK
[pipeline] MuJoCo: OK
[vibe] MediaPipe: OK

============================================
  G1 MISSION CONTROL -- Web Dashboard
============================================
  Mode:    LIVE
  ROMP:    OK
  GMR:     OK
  MuJoCo: OK
  VIBE:    OK
  Camera:  http://localhost:4747/video
  http://localhost:5000
============================================
```

---

## 10. Camera Setup

### DroidCam via ADB (Target PC Setup)

The default configuration uses a phone camera via DroidCam USB:

```bash
# On Linux target machine (in a separate terminal):
adb forward tcp:4747 tcp:4747

# Verify the stream is accessible:
curl -I http://localhost:4747/video
# Should return: HTTP/1.1 200 OK, Content-Type: multipart/x-mixed-replace
```

**Phone orientation**: Hold the phone **vertically** (portrait).
The webapp applies `cv2.ROTATE_90_CLOCKWISE` automatically when `USE_PHONE_CAMERA = True`.

### Switching to Laptop Webcam

```python
# In config.py:
CAMERA_SOURCE    = 0       # 0 = default laptop webcam
USE_PHONE_CAMERA = False   # Don't rotate
```

### Camera Fallback Logic (in `inference_loop()`)

```
1. Try CAMERA_SOURCE (DroidCam URL or integer index)
2. If not opened → try CameraStream(0) (laptop webcam)
3. If neither works → fall back to mock frames
```

This means the inference loop **never crashes** due to a missing camera.

---

## 11. Troubleshooting

### Dashboard loads but streams show nothing after clicking Start

- Check Flask terminal output for errors in `inference_loop` or `vibe_classifier_loop`
- Verify `/health` returns the expected JSON: `curl http://localhost:5000/health`
- If `vibe: false` — MediaPipe or its model file is missing

### `pose_landmarker_heavy.task` not found

```
[vibe] pose_landmarker_heavy.task not found. Searched:
  /home/.../AG/vibe_mocap/vibe_mocap/pose_landmarker_heavy.task
```
**Fix**: The model file must be present at `AGM/vibe_mocap/vibe_mocap/pose_landmarker_heavy.task`.
It is a 30 MB binary — ensure it was committed to git or copied manually.

### `ModuleNotFoundError: No module named 'cv2'`

The Python executable you are running does not have OpenCV installed.
```bash
# Check which python you're using:
which python   (Linux)
where python   (Windows)

# Install for that specific interpreter:
/path/to/correct/python -m pip install opencv-python-headless
```

### `UnicodeEncodeError` when printing to Windows terminal

The app uses only ASCII characters in its output. If you see this with a modified banner, ensure your terminal supports UTF-8:
```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

### ROMP inference is slow / FPS drops to <5

- ROMP requires a CUDA GPU for real-time performance. CPU-only mode is ~2–5 fps.
- Switch to laptop webcam (`CAMERA_SOURCE = 0`) to reduce network overhead.
- Lower `RENDER_WIDTH` / `RENDER_HEIGHT` in config.py.

### Robot floats visually above the floor

Increase `FLOOR_OFFSET` in config.py by 0.01–0.02 increments until the feet touch the grid.

### Port 5000 already in use

```bash
# Linux: find and kill the process
lsof -i :5000
kill -9 <PID>

# Or change the port in config.py:
FLASK_PORT = 5001
```

### GMR raises an IK solver error

This usually means `ROBOT_NAME` is not a valid key in GMR's `ROBOT_XML_DICT`. Check:
```bash
python -c "from general_motion_retargeting.params import ROBOT_XML_DICT; print(list(ROBOT_XML_DICT.keys()))"
```
Use the exact string shown.

---

## 12. Design System Quick Reference

| Class | Purpose |
|---|---|
| `.header` | Top bar with flex layout |
| `.status-chip` | Status indicator pill |
| `.status-dot.active` | Pulsing green dot |
| `.btn-primary` | CTA button (dark default) |
| `.btn-primary.running` | Red state (stop inference) |
| `.mock-banner` | Amber warning bar |
| `.pipeline-info` | Numbered step bar |
| `.dashboard-grid` | 2-column responsive grid |
| `.stream-card` | Panel container |
| `.stream-viewport` | 16:9 feed area |
| `.stream-placeholder` | Idle state content |
| `.metrics-bar` | Auto-fit metrics row |
| `.metric-card` | Individual metric tile |
| `.val-green / .val-cyan / .val-amber / .val-blue / .val-red / .val-purple` | Value colour utilities |
| `.telemetry` | Log section container |
| `.log-entries` | Scrollable log list |

---

*This documentation covers webapp version as of 2026-04-19.*
*Target deployment: `/home/prateek/dhakkan/dl_hackathon/AG/webapp/`*
*Author: Prateek / Antigravity AI pair-programming session*
