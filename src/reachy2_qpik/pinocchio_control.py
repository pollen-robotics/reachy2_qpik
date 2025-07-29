"""Pinocchio IK Control Loop."""

import threading
import time
from collections import deque
from typing import Optional

import numpy as np
import numpy.typing as npt
import pinocchio as pin

from reachy2_qpik.utils import angle_diff, multiturn_safety_check, savitzky_golay


class PinocchioControl:
    """Pinocchio Pose Tracking Control for Reachy2."""

    def __init__(self, node, ik_solver, dt: float = 1 / 500, sg_window: int = 11, sg_order: int = 3):
        """Initialize the class."""
        self.node = node
        self.dt = dt  # [s]
        self.lock = threading.Lock()

        self.q_present = {
            "l_arm": np.zeros(7),  # [rad]
            "r_arm": np.zeros(7),  # [rad]
        }

        self.q_unwrapped = {
            "l_arm": np.zeros(7),  # [rad]
            "r_arm": np.zeros(7),  # [rad]
        }

        self.q_dot_current = {
            "l_arm": np.zeros(7),  # [rad.s⁻¹]
            "r_arm": np.zeros(7),  # [rad.s⁻¹]
        }

        self.target_pose: dict[str, Optional[npt.NDArray[np.float64]]] = {
            "l_arm": None,
            "r_arm": None,
        }

        self.joint_velocity_limits = {
            "l_arm": np.array([7.3] * 7),  # [rad.s⁻¹]
            "r_arm": np.array([7.3] * 7),  # [rad.s⁻¹]
        }

        self.ik_solver = ik_solver
        self.ik_step = ik_solver["r_arm"].dt  # [s]

        self.running = True
        self.emergency_state = ""

        self.sg_window = sg_window
        self.sg_order = sg_order
        self.sg_half = (sg_window - 1) // 2

        self.vel_buffers: dict[str, list[deque[np.float64]]] = {
            "l_arm": [deque(maxlen=sg_window) for _ in range(7)],
            "r_arm": [deque(maxlen=sg_window) for _ in range(7)],
        }

        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._thread.start()

    def _update_joints(self, current_pos: dict[str, float]):
        """Updates the joints values."""
        r_arm_joints = [
            "r_shoulder_pitch",
            "r_shoulder_roll",
            "r_elbow_yaw",
            "r_elbow_pitch",
            "r_wrist_roll",
            "r_wrist_pitch",
            "r_wrist_yaw",
        ]

        l_arm_joints = [
            "l_shoulder_pitch",
            "l_shoulder_roll",
            "l_elbow_yaw",
            "l_elbow_pitch",
            "l_wrist_roll",
            "l_wrist_pitch",
            "l_wrist_yaw",
        ]

        with self.lock:
            ql = np.array([current_pos[name] for name in l_arm_joints])
            qr = np.array([current_pos[name] for name in r_arm_joints])
            self.q_present["l_arm"] = ql
            self.q_present["r_arm"] = qr

    def _control_loop(self):
        """Control Loop for pose tracking."""
        start_time = 0
        loop_count = 0

        while self.running:
            t = time.time()
            loop_count += 1
            for arm in ["l_arm", "r_arm"]:
                with self.lock:
                    q_current = self.q_present[arm].copy()  # [rad]
                    q_dot_current = self.q_dot_current[arm].copy()  # [rad.s⁻¹]
                    target = self.target_pose[arm]

                if target is None:
                    continue

                buffer = self.vel_buffers[arm]
                for j in range(7):
                    buffer[j].append(q_current[j])

                if len(buffer[0]) == self.sg_window:
                    q_dot_smooth = np.zeros(7)
                    for j in range(7):
                        arr = np.array(buffer[j])
                        smooth_sig = savitzky_golay(arr, window_size=self.sg_window, deriv=1, order=self.sg_order, rate=self.ik_step)
                        q_dot_smooth[j] = smooth_sig[self.sg_half]
                    q_dot_current = q_dot_smooth

                target_copy = target.copy()

                q_ddot = self.tick_control(arm, q_current, q_dot_current, target_copy)  # [rad.s⁻²]

                q_dot = q_dot_current + q_ddot * self.ik_step  # [rad.s⁻¹]

                # Speed normalization
                limits = self.joint_velocity_limits[arm]
                scaling = np.abs(q_dot) / limits
                max_scaling = np.max(scaling)

                if max_scaling > 1.0:
                    q_dot = q_dot / max_scaling

                q_updated = pin.integrate(self.ik_solver[arm].model, q_current, q_dot * self.ik_step)  # [rad]

                diffs = np.array([angle_diff(q_updated[i], q_current[i]) for i in range(7)])
                self.q_unwrapped[arm] += diffs

                self.q_unwrapped[arm], emergency, self.emergency_state = multiturn_safety_check(
                    self.q_unwrapped[arm], 6 * np.pi, 6 * np.pi, 6 * np.pi, self.emergency_state
                )

                if emergency:
                    print(f"[EMERGENCY STOP] {arm} joint limits reached.")
                    print(self.emergency_state)
                    self.running = False
                    self.target_pose["l_arm"] = None
                    self.target_pose["r_arm"] = None
                    self.node.publish_joint_commands("l_arm", self.q_present["l_arm"])
                    self.node.publish_joint_commands("r_arm", self.q_present["r_arm"])
                    break

                else:
                    with self.lock:
                        self.q_present[arm] = q_updated

                    self.node.publish_joint_commands(arm, q_updated)

            time.sleep(max(self.dt - (time.time() - t), 0.0))

            if time.time() - start_time >= 1.0:
                # freq = loop_count / (time.time() - start_time)
                # print(f"Frequency: {freq:.2f} Hz")
                loop_count = 0
                start_time = time.time()

    def tick_control(
        self,
        arm: str,
        q_current: npt.NDArray[np.float64],
        q_dot: npt.NDArray[np.float64],
        target_pose: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Update the joint velocities at each tick."""
        try:
            q_ddot = self.ik_solver[arm].compute_acceleration(target_pose, q_current, q_dot)

        except Exception as e:
            print(f"Error in QP computation: {e}")
            q_ddot = np.zeros_like(q_current)

        if q_ddot is None:
            q_ddot = np.zeros_like(q_current)

        return q_ddot

    def set_current_goal(self, arm: str, pose: np.ndarray):
        """Setter method for the current target pose."""
        with self.lock:
            self.target_pose[arm] = pose

    def get_current_goal(self, arm: str) -> Optional[npt.NDArray[np.float64]]:
        """Getter method for the current target pose."""
        with self.lock:
            target = self.target_pose[arm]

        return None if target is None else target.copy()
