# Gesture-Controlled Humanoid Robot (ROBOTIS OP3)
### Real-time hand gesture recognition → ROS 2 → Webots simulation

This project uses a webcam, MediaPipe hand tracking, and a trained PyTorch model to classify hand gestures in real time and translate them into walking commands for a ROBOTIS OP3 humanoid robot simulated in Webots.

---

## Table of Contents
1. [System Architecture](#1-system-architecture)
2. [Prerequisites](#2-prerequisites)
3. [Full Setup Guide (From Scratch)](#3-full-setup-guide-from-scratch)
   - [3.1 Install Webots (Debian — NOT Snap)](#31-install-webots-debian--not-snap)
   - [3.2 Install ROS 2 Jazzy](#32-install-ros-2-jazzy)
   - [3.3 Clone and Build the ROBOTIS OP3 Stack](#33-clone-and-build-the-robotis-op3-stack)
   - [3.4 Set Up the Python AI Environment](#34-set-up-the-python-ai-environment)
4. [How to Run (Every Session)](#4-how-to-run-every-session)
5. [Gesture Reference](#5-gesture-reference)
6. [Manual Testing Commands](#6-manual-testing-commands)
7. [File Reference](#7-file-reference)
8. [Troubleshooting](#8-troubleshooting)
9. [Engineering History & Key Decisions](#9-engineering-history--key-decisions)

---

## 1. System Architecture

```
Webcam
  └── MediaPipe Hand Landmarker (21 landmarks × 3 coords = 63 features)
        └── PyTorch GestureNet (3-layer MLP, 5 classes)
              └── stable_command (majority vote over 7 frames)
                    └── ROS 2 Publisher
                          ├── /robotis/walking/command   (start / stop)
                          └── /robotis/walking/set_params (x/angle amplitude)
                                └── OP3 Manager (C++ ZMP Walking Engine)
                                      └── Webots Simulation (ROBOTIS OP3 body)
```

**Key insight:** The walking physics are handled entirely by the official ROBOTIS C++ walking engine (`op3_manager`). The Python side only needs to publish a `start`/`stop` string and set two amplitude parameters. No custom kinematics.

### Environment
| Component | Version |
|---|---|
| OS | Ubuntu (native, not WSL) |
| ROS 2 | Jazzy |
| Webots | R2025a — **Debian .deb install ONLY** |
| Python | 3.12.3 |
| PyTorch | latest |
| MediaPipe | latest |

---

## 2. Prerequisites

Before starting, make sure your machine has:
- Ubuntu (22.04 or 24.04 recommended)
- A working webcam
- At least 8 GB RAM
- Internet access for downloading packages

---

## 3. Full Setup Guide (From Scratch)

### 3.1 Install Webots (Debian — NOT Snap)

> ⚠️ **Critical:** The Snap version of Webots uses a sandboxed IPC socket at a non-standard path. The ROBOTIS OP3 C++ controller cannot find it, causing a "Process has died" error. You **must** use the native Debian `.deb` package.

**If you have the Snap version installed, remove it first:**
```bash
sudo snap remove webots
```

**Install the Debian version:**
```bash
# Download the .deb from the official Webots release page
wget https://github.com/cyberbotics/webots/releases/download/R2025a/webots_2025a_amd64.deb

# Install it
sudo apt install ./webots_2025a_amd64.deb
```

Verify by running `webots` — it should open from `/usr/local/webots/`.

---

### 3.2 Install ROS 2 Jazzy

If ROS 2 Jazzy is not already installed:
```bash
# Add the ROS 2 apt repository
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
  https://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install ros-jazzy-desktop python3-colcon-common-extensions python3-rosdep
sudo rosdep init
rosdep update
```

---

### 3.3 Clone and Build the ROBOTIS OP3 Stack

This is the most involved step. Follow it exactly — wrong branches or extra packages will cause build failures.

**Create the workspace:**
```bash
mkdir -p ~/gesture_bot_ws/src
cd ~/gesture_bot_ws/src
```

**Clone all required repositories on their `jazzy-devel` branches:**
```bash
# Core ROBOTIS framework and math
git clone -b jazzy-devel https://github.com/ROBOTIS-GIT/ROBOTIS-Framework.git
git clone -b jazzy-devel https://github.com/ROBOTIS-GIT/ROBOTIS-Math.git

# OP3 packages
git clone -b jazzy-devel https://github.com/ROBOTIS-GIT/ROBOTIS-OP3.git
git clone -b jazzy-devel https://github.com/ROBOTIS-GIT/ROBOTIS-OP3-msgs.git
git clone -b jazzy-devel https://github.com/ROBOTIS-GIT/ROBOTIS-OP3-Common.git
git clone -b jazzy-devel https://github.com/ROBOTIS-GIT/ROBOTIS-OP3-Simulations.git
```

> ⚠️ **Required fix:** The `ROBOTIS-OP3-Common` repo contains a `op3_gazebo` folder that conflicts with the simulation package and breaks `colcon build`. Delete it:
> ```bash
> rm -rf ~/gesture_bot_ws/src/ROBOTIS-OP3-Common/op3_gazebo
> ```

**Install system dependencies:**
```bash
sudo apt install ros-jazzy-dynamixel-sdk
cd ~/gesture_bot_ws
rosdep install --from-paths src --ignore-src -r -y
```

**Build:**
```bash
cd ~/gesture_bot_ws
colcon build --symlink-install
```

Build time is approximately 5–10 minutes. It should complete with no errors. Warnings about deprecated CMake syntax are safe to ignore.

**Source the workspace** (add this to your `~/.bashrc` so it applies every session):
```bash
echo "source ~/gesture_bot_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

---

### 3.4 Set Up the Python AI Environment

All AI scripts live in your hackathon directory:
```
~/Desktop/hack_DL/HACK-60-/DL hackathon/
```

**Create and activate a virtual environment:**
```bash
cd ~/Desktop/hack_DL/HACK-60-/"DL hackathon"
python3 -m venv dl_env
source dl_env/bin/activate
```

**Install Python dependencies:**
```bash
pip install torch torchvision mediapipe opencv-python numpy
```

**Verify the following files are present in this directory:**

| File | Purpose |
|---|---|
| `live_bot_dl.py` | Main AI controller script — run this for the demo |
| `gesture_dl_model.pth` | Trained PyTorch model weights |
| `classes.npy` | Gesture class name array (must match model output) |
| `hand_landmarker.task` | MediaPipe hand landmark model file |
| `robot_control.py` | Manual one-shot command tester |
| `walk_forward.py` | Standalone forward walk tester |
| `stop_robot.py` | Standalone stop command sender |

---

## 4. How to Run (Every Session)

You need **three terminals**, opened in this exact order. Wait for each step to complete before opening the next.

---

### Terminal 1 — Webots (Physics Environment)

Loads the OP3 robot body into the simulation world.

```bash
source ~/gesture_bot_ws/install/setup.bash
ros2 launch op3_webots_ros2 robot_launch.py
```

✅ Wait until: Webots opens and the robot model is visible standing on the floor.

---

### Terminal 2 — OP3 Manager (Walking Brain)

Starts the C++ ZMP walking engine that handles all balance physics.

```bash
source ~/gesture_bot_ws/install/setup.bash
ros2 launch op3_manager op3_simulation.launch.py
```

✅ Wait until: The robot in Webots "snaps" into a proper upright standing posture. This means the manager has connected and taken control of the joints.

---

### Terminal 3 — AI Gesture Controller

Activates the webcam, runs the gesture model, and sends commands.

```bash
cd ~/Desktop/hack_DL/HACK-60-/"DL hackathon"
source dl_env/bin/activate
source ~/gesture_bot_ws/install/setup.bash
python3 live_bot_dl.py
```

✅ Wait until: A webcam window appears labelled `Explainable AI Robot Controller`.

**Show your hand to the camera** — the `CMD:` label on screen shows what gesture is detected. The robot will respond.

**To quit:** Press `Q` in the webcam window.

---

## 5. Gesture Reference

| Gesture shown to camera | Class name in model | Robot action |
|---|---|---|
| ✋ Open palm / stop sign | `STOP` | Stops walking, holds position |
| 👆 Pointing forward | `FORWARD` | Walks forward |
| 👈 Pointing left | `TURN_LEFT` | Rotates left on the spot |
| 👉 Pointing right | `TURN_RIGHT` | Rotates right on the spot |
| 🤷 Unrecognised / no hand | `UNKNOWN` | No change (last command held) |

**Confidence threshold:** The model requires >85% softmax confidence. Any prediction below this is treated as `UNKNOWN` and ignored. Commands are also majority-voted over the last 7 frames before being sent to the robot, preventing jitter.

**Transition behaviour:** When switching from any moving command to a new one, the robot automatically stops and waits 2 seconds for the gait cycle to complete before starting the new motion. This prevents the robot from falling due to overlapping momentum.

---

## 6. Manual Testing Commands

Use these to verify each layer of the stack independently, without running the AI script.

**Test the walking engine directly:**
```bash
source ~/gesture_bot_ws/install/setup.bash

# Walk forward
python3 robot_control.py forward

# Turn left
python3 robot_control.py left

# Turn right
python3 robot_control.py right

# Stop
python3 robot_control.py stop
```

**Or use the dedicated scripts:**
```bash
python3 walk_forward.py   # starts walking forward
python3 stop_robot.py     # stops and zeroes all velocity vectors
```

**Verify the ROS topics are active:**
```bash
ros2 topic list | grep robotis
# Should show: /robotis/walking/command, /robotis/walking/set_params, etc.
```

**Publish a command manually:**
```bash
ros2 topic pub --once /robotis/walking/command std_msgs/msg/String "data: 'start'"
ros2 topic pub --once /robotis/walking/command std_msgs/msg/String "data: 'stop'"
```

---

## 7. File Reference

### AI Hackathon Directory
```
~/Desktop/hack_DL/HACK-60-/DL hackathon/
├── live_bot_dl.py          ← MAIN SCRIPT — run this for the demo
├── gesture_dl_model.pth    ← Trained PyTorch weights (do not delete)
├── classes.npy             ← Class name array (must match model exactly)
├── hand_landmarker.task    ← MediaPipe model file
├── robot_control.py        ← Manual command tester (CLI)
├── walk_forward.py         ← Standalone forward walk test
├── stop_robot.py           ← Standalone stop test
├── train_dl_model.py       ← Training script (not needed at runtime)
├── record_data.py          ← Dataset recording script
└── normalise_data.py       ← Data normalisation script
```

### ROS 2 Workspace
```
~/gesture_bot_ws/src/
├── ROBOTIS-Framework/      ← Core servo control framework
├── ROBOTIS-Math/           ← ZMP and kinematics math library
├── ROBOTIS-OP3/            ← Main OP3 ROS 2 packages (op3_manager lives here)
├── ROBOTIS-OP3-msgs/       ← Custom message/service definitions (WalkingParam etc.)
├── ROBOTIS-OP3-Common/     ← Shared configs and description files
└── ROBOTIS-OP3-Simulations/← Webots world and launch files
```

### Key ROS 2 Topics
| Topic | Type | Purpose |
|---|---|---|
| `/robotis/walking/command` | `std_msgs/String` | `"start"` or `"stop"` |
| `/robotis/walking/set_params` | `WalkingParam` | Set x/angle amplitude |
| `/robotis/walking/get_params` | Service | Fetch current stable params |
| `/robotis/enable_ctrl_module` | `std_msgs/String` | Activate `"walking_module"` |

### Walking Parameter Values
| Command | `x_move_amplitude` | `angle_move_amplitude` |
|---|---|---|
| FORWARD | `0.025` m | `0.0` rad |
| TURN_LEFT | `0.0` m | `+0.2` rad |
| TURN_RIGHT | `0.0` m | `-0.2` rad |
| STOP | *(send "stop" command)* | — |

---

## 8. Troubleshooting

### Robot does not snap to standing posture when Terminal 2 launches
- Webots must be **fully loaded** (Terminal 1) before launching `op3_manager`.
- Check that you're using the **Debian install** of Webots, not Snap.
- Verify with: `which webots` — should return `/usr/local/webots/webots`, not `/snap/...`

### "Process has died" error in Terminal 2
Almost always caused by the Snap version of Webots. The C++ manager communicates via `/tmp/` IPC sockets, which Snap's sandbox blocks.
```bash
sudo snap remove webots
# Then re-install the .deb version as in Section 3.1
```

### `colcon build` fails with "Finddynamixel_sdk" error
```bash
sudo apt install ros-jazzy-dynamixel-sdk
cd ~/gesture_bot_ws
colcon build --symlink-install
```

### `colcon build` fails with "Findcatkin" error
```bash
cd ~/gesture_bot_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
```

### Build fails due to duplicate package definition
Make sure you deleted the conflicting Gazebo folder:
```bash
rm -rf ~/gesture_bot_ws/src/ROBOTIS-OP3-Common/op3_gazebo
colcon build --symlink-install
```

### Webcam window opens but turns always show `CMD: TURN_LEFT` or `TURN_RIGHT` with no robot response
This was the main bug in the original script — a mismatch between model class names (`TURN_LEFT`/`TURN_RIGHT`) and the controller's if/elif checks (`LEFT`/`RIGHT`). The fixed `live_bot_dl.py` in this repo resolves it.

### Robot falls over when changing direction
The 2-second stabilisation cooldown in `execute_gesture()` should prevent this. If it still falls, increase the sleep duration:
```python
time.sleep(2.0)  # increase to 3.0 if needed
```

### `op3_walking_module_msgs` not found when running `live_bot_dl.py`
You forgot to source the workspace in Terminal 3:
```bash
source ~/gesture_bot_ws/install/setup.bash
```
Add this to your `~/.bashrc` to make it permanent.

### MediaPipe / PyTorch import errors in virtual environment
Ensure you activated the virtual environment before running:
```bash
source dl_env/bin/activate
```

---

## 9. Engineering History & Key Decisions

This section documents what was tried, what failed, and why the current architecture was chosen. Useful context if you need to extend or debug the project.

### What was abandoned and why

**Custom sinusoidal gait (Failed):** Driving leg joints with `sin(t)` waves caused the robot to fall sideways immediately. Root cause: no lateral weight shift before lifting a foot. When any foot leaves the ground, the centre of mass remains centred — outside the support polygon — and the robot topples.

**Sinusoidal gait + hip roll compensation (Failed):** Added `PelvR`/`PelvL` driven by `cos(φ)` to shift the CoM laterally before swing. Root cause of failure: the sign convention of the pelvic roll joints in this specific URDF is unknown without empirical testing, and the Webots documentation explicitly states: *"In simulation, lateral balance does not work as expected. It is recommended to set `balance_hip_roll_gain` and `balance_ankle_roll_gain` to 0.0."* Hip-roll-based lateral balance is fundamentally unreliable in Webots physics for this robot.

**Webots `.motion` file API (Abandoned — files missing):** The Motion file playback API (`from controller import Motion`) would have been ideal, but the Snap installation did not include the pre-built `.motion` files for the OP3, and writing calibrated motion files from scratch requires knowing exact joint angle sign conventions.

### What works and why

**Official ROBOTIS C++ Walking Engine:** The `op3_manager` package implements a full ZMP-based balance algorithm tuned by ROBOTIS engineers for this exact robot. It handles the complex balance physics internally. Python only needs to send a `start`/`stop` string and two amplitude scalars (`x_move_amplitude`, `angle_move_amplitude`). This is the correct abstraction boundary.

**Debian Webots install:** The C++ walking engine communicates with Webots via IPC sockets in `/tmp/`. The Snap sandbox intercepts and redirects these to a non-standard path that the ROS nodes cannot find, causing immediate crashes. The Debian `.deb` install uses standard paths and works correctly.

**`jazzy-devel` branches:** The default/master branches of all ROBOTIS repositories are for ROS 1. Using them with ROS 2 Jazzy produces cascading build failures. All six repositories must be on their `jazzy-devel` branch.

**Sticky command + 2-second cooldown:** Without this, receiving a new gesture while the robot is mid-stride causes two gait cycles to overlap, resulting in the robot falling. The cooldown waits for the current foot to complete its arc and both feet to be planted before starting the new motion.
