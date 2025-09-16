"""Pinocchio IK Speed Control Loop."""

import threading
import time
from typing import Optional

import numpy as np
import numpy.typing as npt
import pinocchio as pin

from reachy2_qpik.utils import limit_orbita3d_joints_wrist, multiturn_safety_check


class Reachy2SpeedControl:
    """Real-time speed-based pose tracking control for Reachy2.

    This class continuously updates the joint positions using an IK solver,
    applies velocity normalization and joint limits, and publishes commands
    to the robot through pollen_kdl_kinematics_node.py.
    """

    def __init__(self, node, ik_solver, dt: float = 1 / 500):
        """Initialize the Reachy2SpeedControl loop.

        Args:
            node: ROS2 node or interface used to send joint commands.
            ik_solver (dict): Dictionary of IK solvers for each arm
                (keys: "l_arm", "r_arm").
            dt (float, optional): Control loop period in seconds.
                Defaults to 1/500 (500 Hz).
        """
        self.node = node
        self.dt = dt  # [s]
        self.lock = threading.Lock()

        self.ik_solver = ik_solver
        self.ik_step = ik_solver["r_arm"].dt  # [s]

        self.q_present = {
            "l_arm": np.zeros(7),  # [rad]
            "r_arm": np.zeros(7),  # [rad]
        }

        self.target_pose: dict[str, Optional[npt.NDArray[np.float64]]] = {
            "l_arm": None,
            "r_arm": None,
        }

        self.joint_velocity_limits = {
            "l_arm": np.array([7.3] * 7),  # [rad.s⁻¹]
            "r_arm": np.array([7.3] * 7),  # [rad.s⁻¹]
        }

        self.running = True
        self.emergency_state = ""

        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._thread.start()

    def _update_joints(self, current_pos: dict[str, float]):
        """Update the joint positions from the current state.

        Args:
            current_pos (dict[str, float]): Dictionary mapping joint names
                to their current positions in radians.
        """
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
        """Run the main control loop for pose tracking.

        This loop:
            - Reads the current joint states.
            - Computes velocities using the IK solver.
            - Applies velocity limits and integrates joint velocities.
            - Publishes updated joint positions to the robot.
            - Stops if multiturns are detected.
        """
        start_time = 0
        # loop_count = 0

        while self.running:
            t = time.time()
            loop_count += 1
            for arm in ["l_arm", "r_arm"]:
                with self.lock:
                    q_current = self.q_present[arm]  # [rad]
                    target = self.target_pose[arm]

                if target is None:
                    continue

                target_copy = target.copy()

                q_dot = self.tick_control(arm, q_current, target_copy)  # [rad.s⁻¹]

                # Speed normalization
                limits = self.joint_velocity_limits[arm]
                scaling = np.abs(q_dot) / limits
                max_scaling = np.max(scaling)

                if max_scaling > 1.0:
                    q_dot = q_dot / max_scaling

                q = pin.integrate(self.ik_solver[arm].model, q_current, q_dot * self.ik_step)  # [rad]
                q = np.array(limit_orbita3d_joints_wrist(list(q), 74.17649320975901))

                q, emergency, self.emergency_state = multiturn_safety_check(
                    q, 4 * np.pi, 4 * np.pi, 4 * np.pi, self.emergency_state
                )

                if emergency:
                    print(f"[EMERGENCY STOP] {arm} joint limits reached.")
                    print(self.emergency_state)
                    self.running = False
                    self.target_pose["l_arm"] = None
                    self.target_pose["r_arm"] = None
                    self.node.publish_joint_commands("l_arm", q)
                    self.node.publish_joint_commands("r_arm", q)
                    break

                else:
                    with self.lock:
                        self.q_present[arm] = q

                self.node.publish_joint_commands(arm, q)

            time.sleep(max(self.dt - (time.time() - t), 0.0))

            if time.time() - start_time >= 1.0:
                # freq = loop_count / (time.time() - start_time)
                # print(f"Frequency: {freq:.2f} Hz")
                # loop_count = 0
                start_time = time.time()

    def tick_control(
        self, arm: str, q_current: npt.NDArray[np.float64], target_pose: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute joint velocities for the current control tick.

        Args:
            arm (str): Arm name ("l_arm" or "r_arm").
            q_current (numpy.ndarray): Current joint positions [rad].
            target_pose (numpy.ndarray): Desired end-effector pose (4x4 SE3 matrix).

        Returns:
            numpy.ndarray: Computed joint velocities [rad.s⁻¹].
        """
        try:
            q_dot = self.ik_solver[arm].compute_velocity(target_pose, q_current)

        except Exception:
            q_dot = np.zeros_like(q_current)

        return q_dot

    def set_current_goal(self, arm: str, pose: np.ndarray):
        """Set a new target pose for the specified arm.

        Args:
            arm (str): Arm to control ("l_arm" or "r_arm").
            pose (numpy.ndarray): Target end-effector pose (4x4 SE3 matrix).
        """
        with self.lock:
            self.target_pose[arm] = pose

    def get_current_goal(self, arm: str) -> Optional[npt.NDArray[np.float64]]:
        """Get the current target pose for the specified arm.

        Args:
            arm (str): Arm to query ("l_arm" or "r_arm").

        Returns:
            numpy.ndarray | None: Current target pose (4x4 SE3 matrix) or None if no target is set.
        """
        with self.lock:
            target = self.target_pose[arm]

        return None if target is None else target.copy()
