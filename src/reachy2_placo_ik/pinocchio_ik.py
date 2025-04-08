"""Reachy2 Pinocchio IK class"""

import os

import numpy as np
import numpy.typing as npt
import pinocchio as pin
from numpy.linalg import norm, solve
from scipy.spatial.transform import Rotation as R


class PinocchioIK:
    def __init__(
        self,
        urdf_path: str,
        arm: str = "r_arm",
        locked_joints: list[str] = None,
    ) -> None:
        robot = pin.RobotWrapper.BuildFromURDF(urdf_path, os.path.dirname(urdf_path))

        if locked_joints is None:
            locked_joints = self.default_locked_joints(arm)

        robot = robot.buildReducedRobot(locked_joints)

        self.robot = robot
        self.model, self.data = robot.model, robot.data

        self.arm = arm
        self.ee_frame = f"{arm}_tip"
        self.ee_frame_id = self.model.getFrameId(self.ee_frame)
        self.joint_id = self.model.frames[self.ee_frame_id].parent

        self.eps = 1e-4  # Error precision
        self.IT_MAX = 1000  # Maximum number of iteration
        self.DT = 5e-1  # Time step
        self.damp = 1e-12  # Damping factor
        self.v_max = 7e-1  # Maximum velocity update

    def default_locked_joints(self, arm: str) -> list[str]:
        """List of the default joints to lock before computation"""
        shared_joints = [
            "tripod_joint",
            "l_hand_finger",
            "l_hand_finger_proximal",
            "l_hand_finger_distal",
            "l_hand_finger_proximal_mimic",
            "l_hand_finger_distal_mimic",
            "left_bar_joint_mimic",
            "left_bar_prism_joint_mimic",
            "neck_roll",
            "neck_pitch",
            "neck_yaw",
            "antenna_left",
            "antenna_right",
            "r_hand_finger",
            "r_hand_finger_proximal",
            "r_hand_finger_distal",
            "r_hand_finger_proximal_mimic",
            "r_hand_finger_distal_mimic",
            "right_bar_joint_mimic",
            "right_bar_prism_joint_mimic",
        ]

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

        if arm == "r_arm":
            return shared_joints + l_arm_joints
        elif arm == "l_arm":
            return shared_joints + r_arm_joints
        else:
            return shared_joints

    def inverse_kinematics(self, goal_pose: npt.NDArray[np.float64], current_joints: npt.NDArray[np.float64]) -> tuple:
        """Get the joints from the Inverse Kinematics"""
        R_goal = goal_pose[:3, :3]
        p_goal = goal_pose[:3, 3]
        oMdes_base = pin.SE3(R_goal, p_goal)

        T_baselink_torso = pin.SE3(np.eye(3), np.array([0.010, 0.000, -0.996]))
        oMdes = T_baselink_torso.inverse() * oMdes_base

        if current_joints is None:
            q = pin.neutral(self.model)
        else:
            q = current_joints.copy()

        success = False
        state = ""
        i = 0

        while i < self.IT_MAX:
            pin.forwardKinematics(self.model, self.data, q)
            iMd = self.data.oMi[self.joint_id].actInv(oMdes)
            err = pin.log(iMd).vector

            if norm(err) < self.eps:
                success = True
                state = f"Convergence reached in {i} iterations."
                break

            # Closed-Loop Inverse Kinematics
            J = pin.computeJointJacobian(self.model, self.data, q, self.joint_id)
            J = -np.dot(pin.Jlog6(iMd.inverse()), J)

            v = -J.T.dot(solve(J.dot(J.T) + self.damp * np.eye(6), err))
            v = np.clip(v, -self.v_max, self.v_max)

            q = pin.integrate(self.model, q, v * self.DT)

            # Newton-Raphson
            # dq = np.dot(np.linalg.pinv(J), err)
            # dq = np.clip(dq, -self.v_max, self.v_max)

            # q = pin.integrate(self.model, q, -dq*self.DT)

            i += 1

        if not success:
            state = "Convergence not reached with maximum iterations."

        return q, success, state
