"""Reachy2 Pinocchio Inverse Kinematics class."""

import os
from typing import Optional

import numpy as np
import numpy.typing as npt
import pinocchio as pin
from numpy.linalg import norm, solve


class PinocchioIK:
    """Pinocchio IK class for Reachy2."""

    def __init__(
        self,
        urdf_path: str,
        arm: str = "r_arm",
        locked_joints: Optional[list[str]] = None,
    ) -> None:
        """Initialize the class."""
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
        self.DT = 5e-1  # Time step
        self.damp = 4.5e-3  # Damping factor
        self.IT_MAX = 1

    def default_locked_joints(self, arm: str) -> list[str]:
        """List of the default joints to lock before computation."""
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

    def inverse_kinematics(
        self, goal_pose: npt.NDArray[np.float64], current_joints: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], bool, str]:
        """Get the joints from the Inverse Kinematics."""
        R_goal = goal_pose[:3, :3]
        p_goal = goal_pose[:3, 3]
        oMdes_torso = pin.SE3(R_goal, p_goal)

        q_neutral = pin.neutral(self.model)
        pin.forwardKinematics(self.model, self.data, q_neutral)
        pin.updateFramePlacements(self.model, self.data)

        T_baselink_torso = self.data.oMf[self.model.getFrameId("torso")].copy()

        oMdes = T_baselink_torso * oMdes_torso

        if current_joints is None:
            q = pin.neutral(self.model)
        else:
            q = current_joints.copy()

        success = False
        state = ""

        i = 0
        while i < self.IT_MAX:
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            iMd = self.data.oMi[self.joint_id].actInv(oMdes)
            err = pin.log(iMd).vector

            if norm(err) < self.eps:
                success = True
                state = "Convergence reached."
                break

            J = pin.computeFrameJacobian(self.model, self.data, q, self.ee_frame_id, pin.ReferenceFrame.LOCAL)
            J = -np.dot(pin.Jlog6(iMd.inverse()), J)

            # Velocity test
            # dq = np.zeros(7)
            # dq[6] = 1.0

            # twist = J.dot(dq)
            # v, w = twist[:3], twist[3:]

            # print(f"  Linear part   v = {v}")
            # print(f"  Angular part  ω = {w}\n")
            # print(f"  |v| = {norm(v):.5f},  |ω| = {norm(w):.5f}\n")

            # print(J)

            # Closed-Loop Inverse Kinematics
            v = -J.T.dot(solve(J.dot(J.T) + self.damp * np.eye(6), err))
            q = pin.integrate(self.model, q, v * self.DT)

            if not success:
                state = "Convergence not reached."

            i += 1

        # Forward Kinematics obtained by Pinocchio
        # print(self.arm)
        # pin.forwardKinematics(self.model, self.data, q)
        # pin.updateFramePlacements(self.model, self.data)
        # final_ee_pose = T_baselink_torso.inverse() * self.data.oMf[self.ee_frame_id]
        # print(final_ee_pose)

        return q, True, state
