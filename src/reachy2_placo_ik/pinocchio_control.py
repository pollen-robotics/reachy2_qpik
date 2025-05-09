import threading
import time

import numpy as np
import numpy.typing as npt
import pinocchio as pin


class PinocchioControl:
    """Pinocchio Tracking Control for Reachy2."""

    def __init__(self, node, ik_solver, dt: float = 1 / 500):
        """Initialize the class."""
        self.node = node
        self.dt = dt  # [s]
        self.lock = threading.Lock()

        self.q_present = {
            "l_arm": np.zeros(7),  # [rad]
            "r_arm": np.zeros(7),  # [rad]
        }

        self.target_pose = {
            "l_arm": None,
            "r_arm": None,
        }

        self.ik_solver = ik_solver

        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._thread.start()

    def _update_joints(self, current_pos: npt.NDArray[np.float64]):
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

        while True:
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
                q_updated = pin.integrate(self.ik_solver[arm].model, q_current, q_dot * 0.17)  # [rad]
                # q_updated = q_current + q_dot * self.dt  # [rad]

                with self.lock:
                    self.q_present[arm] = q_updated

                self.node.publish_joint_commands(arm, q_updated)

            time.sleep(max(self.dt - (time.time() - t), 0.0))

            if time.time() - start_time >= 1.0:
                freq = loop_count / (time.time() - start_time)
                print(f"Frequency: {freq:.2f} Hz")
                loop_count = 0
                start_time = time.time()

    def tick_control(
        self, arm: str, q_current: npt.NDArray[np.float64], target_pose: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Updates the joint velocities at each tick."""
        try:
            q_dot = self.ik_solver[arm].compute_velocity(target_pose, q_current)

        except Exception:
            q_dot = np.zeros_like(q_current)

        return q_dot

    def set_current_goal(self, arm: str, pose: np.ndarray):
        """Setter method for the current target pose"""
        with self.lock:
            self.target_pose[arm] = pose

    def get_current_goal(self, arm: str) -> npt.NDArray[np.float64]:
        """Getter method for the current target pose."""
        with self.lock:
            target = self.target_pose[arm]

        return None if target is None else target.copy()
