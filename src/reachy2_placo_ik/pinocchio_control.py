import threading
import time

import numpy as np


class PinocchioControl:
    """Pinocchio Tracking Control for Reachy2."""

    def __init__(self, node, ik_solver, dt: float = 1 / 500):
        print("Yepee")
        """Initialize the class"""
        self.node = node
        self.dt = dt
        self.lock = threading.Lock()

        self.q_present = {
            "l_arm": np.zeros(7),
            "r_arm": np.zeros(7),
        }

        self.target_pose = {
            "l_arm": None,
            "r_arm": None,
        }

        self.ik_solver = ik_solver

        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._thread.start()

    def _update_joints(self, current_pos):
        """ """
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
        while True:
            t = time.time()
            for arm in ["l_arm", "r_arm"]:
                with self.lock:
                    q_current = self.q_present[arm]
                    target = self.target_pose[arm]

                if target is None:
                    continue

                target_copy = target.copy()

                dq = self.tick_control(arm, q_current, target_copy)
                q_updated = q_current + dq * self.dt

                with self.lock:
                    self.q_present[arm] = q_updated

                self.node.publish_joint_commands(arm, q_updated)

            time.sleep(max(self.dt - (time.time() - t), 0.0))

    def tick_control(self, arm: str, q_current, target_pose):
        """Updates the joint velocities at each tick."""
        try:
            dq = self.ik_solver[arm].compute_velocity(target_pose, q_current)

        except Exception:
            dq = np.zeros_like(q_current)

        return dq

    def set_current_goal(self, arm: str, pose: np.ndarray):
        """Setter method for the current target pose"""
        with self.lock:
            self.target_pose[arm] = pose

    def get_current_goal(self, arm):
        """Getter method for the current target pose."""
        with self.lock:
            target = self.target_pose[arm]

        return None if target is None else target.copy()
