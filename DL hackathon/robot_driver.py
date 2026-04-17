"""
robot_driver.py
---------------
Webots external controller for ROBOTIS OP3.
Subscribed topic : /gesture_command  (std_msgs/String)
Valid commands   : FORWARD | STOP | TURN_LEFT | TURN_RIGHT | DAB

State machine
─────────────
  STOPPED        → all motors parked at rest pose
  FORWARD        → straight sinusoidal walking gait
  TURNING_LEFT   → walking gait with left differential (curves left)
  TURNING_RIGHT  → walking gait with right differential (curves right)

Turn commands keep the robot moving if it was already moving, and
continue in that new heading after the command clears.

IMPORTANT — Verify motor names first
──────────────────────────────────────
If the robot doesn't move, add this one-liner to `init()` and check the
Webots console output against the names used in GAIT_JOINTS below:

    print(list(self.__robot.getDeviceList()))

"""

import math
import rclpy
from std_msgs.msg import String

# ── Timestep (ms) ── must match your world's basicTimeStep
TIMESTEP_MS = 32

# ── Gait parameters (tune these to taste) ───────────────────────────────
GAIT_FREQ     = 1.2    # walking cycles per second
HIP_AMP       = 0.40   # hip pitch swing amplitude  (radians)
KNEE_AMP      = 0.30   # knee flexion amplitude     (radians)
ANKLE_AMP     = 0.20   # ankle compensation amplitude
SHOULDER_AMP  = 0.20   # arm counter-swing amplitude
TURN_BIAS     = 0.25   # extra amplitude on outer leg when turning

# ── Motor name mapping (RIGHT side, LEFT side) ──────────────────────────
# If any name is wrong, the getDevice() call returns None and you'll see
# a clear error message. Fix the string to match your .wbt motor names.
GAIT_JOINTS = {
    # (right_motor_name, left_motor_name)
    'hip_pitch'  : ('LegUpperR',  'LegUpperL'),
    'knee'       : ('LegLowerR',  'LegLowerL'),
    'ankle_pitch': ('AnkleR',     'AnkleL'),
    'shoulder'   : ('ShoulderR',  'ShoulderL'),
    'hip_yaw'    : ('PelvYR',     'PelvYL'),
    'hip_roll'   : ('PelvR',      'PelvL'),
}

# Rest positions (all zeros keeps OP3 in default neutral stance)
REST_POSITIONS = {
    'LegUpperR': 0.0, 'LegUpperL': 0.0,
    'LegLowerR': 0.0, 'LegLowerL': 0.0,
    'AnkleR'   : 0.0, 'AnkleL'   : 0.0,
    'ShoulderR': 0.0, 'ShoulderL': 0.0,
    'PelvYR'   : 0.0, 'PelvYL'   : 0.0,
    'PelvR'    : 0.0, 'PelvL'    : 0.0,
    'Neck'     : 0.0,
}

# ── Robot state constants ────────────────────────────────────────────────
STATE_STOPPED       = 'STOPPED'
STATE_FORWARD       = 'FORWARD'
STATE_TURNING_LEFT  = 'TURNING_LEFT'
STATE_TURNING_RIGHT = 'TURNING_RIGHT'
STATE_DAB           = 'DAB'


class RobotDriver:

    # ────────────────────────────────────────────────────────────────────
    def init(self, webots_node, properties):
        self.__robot = webots_node.robot

        # ── Head ────────────────────────────────────────────────────────
        self.__head = self.__robot.getDevice('Neck')

        # ── Leg & arm motors ────────────────────────────────────────────
        self.__motors = {}
        for group, (r_name, l_name) in GAIT_JOINTS.items():
            for name in (r_name, l_name):
                device = self.__robot.getDevice(name)
                if device is None:
                    print(f'⚠️  Motor "{name}" not found. '
                          f'Run print(list(robot.getDeviceList())) to check names.')
                else:
                    self.__motors[name] = device

        # ── ROS 2 ────────────────────────────────────────────────────────
        if not rclpy.ok():
            rclpy.init(args=None)
        self.__node = rclpy.create_node('gesture_brain')
        self.__node.create_subscription(
            String,
            '/gesture_command',
            self.__command_callback,
            10
        )

        # ── Internal state ───────────────────────────────────────────────
        self.__state       = STATE_STOPPED
        self.__gait_phase  = 0.0          # radians, advances each step()
        self.__dt          = TIMESTEP_MS / 1000.0  # seconds per step

        print('✅ ROBOT READY — Listening on /gesture_command')
        print(f'   Known motors: {list(self.__motors.keys())}')

    # ────────────────────────────────────────────────────────────────────
    def __command_callback(self, msg):
        cmd = msg.data.upper().strip()
        prev = self.__state

        if cmd == 'FORWARD':
            self.__state = STATE_FORWARD

        elif cmd == 'STOP':
            self.__state = STATE_STOPPED

        elif cmd == 'TURN_LEFT':
            # Keep locomotion going; just change heading direction
            if self.__state == STATE_STOPPED:
                self.__state = STATE_TURNING_LEFT   # turn in place
            else:
                self.__state = STATE_TURNING_LEFT   # walk + turn

        elif cmd == 'TURN_RIGHT':
            if self.__state == STATE_STOPPED:
                self.__state = STATE_TURNING_RIGHT
            else:
                self.__state = STATE_TURNING_RIGHT

        elif cmd == 'DAB':
            self.__state = STATE_DAB

        if self.__state != prev:
            print(f'  State: {prev} → {self.__state}')

    # ────────────────────────────────────────────────────────────────────
    def step(self):
        """Called every simulation timestep by Webots."""
        rclpy.spin_once(self.__node, timeout_sec=0)

        if self.__state == STATE_STOPPED:
            self.__park_motors()

        elif self.__state == STATE_FORWARD:
            self.__walk(left_bias=0.0, right_bias=0.0)
            self.__advance_phase()

        elif self.__state == STATE_TURNING_LEFT:
            # Outer (right) leg gets more amplitude → body curves left
            self.__walk(left_bias=-TURN_BIAS, right_bias=TURN_BIAS)
            self.__advance_phase()

        elif self.__state == STATE_TURNING_RIGHT:
            # Outer (left) leg gets more amplitude → body curves right
            self.__walk(left_bias=TURN_BIAS, right_bias=-TURN_BIAS)
            self.__advance_phase()

        elif self.__state == STATE_DAB:
            # Placeholder — DAB/mimic mode will be implemented separately
            self.__park_motors()

    # ── Private helpers ──────────────────────────────────────────────────

    def __advance_phase(self):
        """Increment the gait phase by one timestep."""
        self.__gait_phase += 2.0 * math.pi * GAIT_FREQ * self.__dt

    def __park_motors(self):
        """Return all motors to the rest (neutral) pose."""
        for name, motor in self.__motors.items():
            motor.setPosition(REST_POSITIONS.get(name, 0.0))
        if self.__head:
            self.__head.setPosition(0.0)

    def __set_motor(self, name, position):
        """Safe motor setter — silently skips if the device wasn't found."""
        if name in self.__motors:
            self.__motors[name].setPosition(position)

    def __walk(self, left_bias=0.0, right_bias=0.0):
        """
        Sinusoidal bipedal walking gait.

        Phase conventions
        ─────────────────
        Right leg leads at phase = 0  (sin > 0  → swing forward)
        Left  leg leads at phase = π  (sin < 0  → swing forward)

        left_bias / right_bias
        ──────────────────────
        Positive bias → that leg swings more (body curves away from it).
        Used for differential turning.
        """
        phi   = self.__gait_phase
        s_r   =  math.sin(phi)          # right leg phase
        s_l   = -math.sin(phi)          # left leg phase (π offset)

        r_hip_amp = HIP_AMP   + right_bias
        l_hip_amp = HIP_AMP   + left_bias

        # ── Hip pitch (forward swing) ────────────────────────────────
        self.__set_motor('LegUpperR',  s_r * r_hip_amp)
        self.__set_motor('LegUpperL',  s_l * l_hip_amp)

        # ── Knee (flex on back-swing to clear ground) ────────────────
        # Knee bends (positive) when leg is on back-swing (sin < 0)
        self.__set_motor('LegLowerR',  max(0.0, -s_r) * KNEE_AMP)
        self.__set_motor('LegLowerL',  max(0.0, -s_l) * KNEE_AMP)

        # ── Ankle compensation (keep sole flat) ──────────────────────
        self.__set_motor('AnkleR',    -s_r * ANKLE_AMP)
        self.__set_motor('AnkleL',    -s_l * ANKLE_AMP)

        # ── Arm counter-swing (balance) ──────────────────────────────
        # Arms swing opposite to legs (natural human gait)
        self.__set_motor('ShoulderR', -s_r * SHOULDER_AMP)
        self.__set_motor('ShoulderL', -s_l * SHOULDER_AMP)

        # ── Hip yaw for turning ──────────────────────────────────────
        # Apply a small yaw offset to the stance leg
        yaw_offset = (right_bias - left_bias) * 0.3
        self.__set_motor('PelvYR',  yaw_offset)
        self.__set_motor('PelvYL', -yaw_offset)